"""
This file patches the flamingo gated cross attention into the T5 decoder.
"""
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.models.t5.modeling_t5 import (T5Block, T5LayerCrossAttention,
                                                T5LayerFF,
                                                T5LayerSelfAttention)

from .gated_xattn import HijackedLMBlock

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


@dataclass
class FlamingoConfig:
    """Configuration class to store the configuration of a `FlamingoT5` model."""
    xattn_every: int = 1


class FlamingoBaseModel(ABC, PreTrainedModel):
    """ 
    abstract class, which is inherited by FlamingoGPT2 and FlamingoOPT.
    This class provides the core functionalities of Flamingo: the forward() function,
    setting up the resampler and hijacking the LM layers with GatedXAttn layers.
    """

    config: FlamingoConfig
    lm: PreTrainedModel
    lm_head: nn.Linear

    config_class = FlamingoConfig

    def __init__(self, config: FlamingoConfig, suppress_warnings=True):
        assert isinstance(config, FlamingoConfig)
        super().__init__(config)

    def _init_layers(self, lm_layers: nn.ModuleList):
        """
        call during init of the subclass.
        careful, this method will modify the LM layers!
        """
        for i, lm_layer in enumerate(lm_layers):
            if i % self.config.xattn_every != 0: 
                continue

            lm_layers[i] = HijackedLMBlock(
                lm_layer,
                dim=self.config.dim,
                dim_visual=self.config.dim_visual,
                dim_head=self.config.xattn_dim_head,
                heads=self.config.xattn_heads,
                ff_mult=self.config.xattn_ff_mult,
                act=self.config.xattn_act,
                n_visual=self.config.resampler_num_latents
            )

    @abstractmethod
    def get_modified_layers(self) -> List[HijackedLMBlock]:
        raise NotImplementedError

    def freeze_vm(self):
        """freeze vision model """
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        """ freeze weights of the language model.

        (!) does not freeze token embedding matrix and gated xattn layers
        """

        for param in self.lm.parameters():
            param.requires_grad = False

        # lm_head shares weights with the embeddings so no need to unfreeze that as well
        self.lm.get_input_embeddings().weight.requires_grad = True

        for xattn in self.get_modified_layers():
            for param in xattn.xattn_block.parameters():
                param.requires_grad = True

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        """ include weights in the state dict if they have requires_grad = True"""

        trainable_param_names = [
            w for w, t in self.named_parameters() if t.requires_grad]
        return {k: v for k, v in self.state_dict().items() if k in trainable_param_names}

    def parameters_trainable(self):
        """Access the trainable parameters, e.g. useful for the optimizer and gradient clipping. 

        example: optimizer = AdamW(model.parameters_trainable(), lr=args.lr)
        make sure to call freeze_lm() first! 
        """
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def encode_resample_visuals(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pass pixel values through vision encoder and perceiver resampler.

        Args:
            pixel_values (torch.Tensor): accepted shapes:
                (N c h w)       one batch, multiple images
                (b N c h w)     multiple batches, multiple images
                (b N T c h w)   multiple batches, multiple images, multiple frames

        Returns:
            (torch.Tensor): shape (b N q d)
        """

        if pixel_values.ndim == 4:            
            # (N c h w)
            b, N, T = 1, pixel_values.size(0), 1
        
        elif pixel_values.ndim == 5:       
            # (b N c h w)
            b, N, T = *pixel_values.shape[:2], 1
            pixel_values = rearrange(pixel_values, 'b N c h w -> (b N) c h w')

        elif pixel_values.ndim == 6:         
            # (b N T c h w) -> (b N T v d)
            b, N, T = pixel_values.shape[:3]
            pixel_values = rearrange(pixel_values, 'b N T c h w -> (b N T) c h w')
        else:
            raise ValueError('pixel_values must have ndim 5 or 6!')

        with torch.no_grad():
            visual_features = self.vision_encoder(pixel_values).last_hidden_state         # (b N T) v d

        # perceiver resampler
        # (only need to do if kv of the xattn layers were not calculated yet.)
        # resample visual features ((b N T) v d) -> (b N T q d)
        visual_features = rearrange(visual_features, '(b N T) v d -> (b N) T v d', b=b, N=N, T=T)
        visual_features = self.resampler(visual_features)

        # T is gone at this point
        visual_features = rearrange(visual_features, '(b N) q d -> b N q d', b=b, N=N)
        
        return visual_features
        
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        media_locations: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:
        """Flamingo forward pass

        Most of the parameters are inspired by huggingface language model implementations, so this doc may be informative:
        https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model.forward

        Args:
            input_ids (Tensor | None):         shape (n_batch, n_tokens). the tokenized input text
            attention_mask (Tensor | None):    shape (n_batch, n_tokens). 
                Mask as produced by the tokenizer. Required when a batch of input strings are tokenized and thus padded at the end.
                Then this will indicate the locations of 'real' tokens vs. the location of 'pad' tokens.
            media_locations (Tensor | None):   shape (n_batch, n_tokens).
                indicates the locations of the starts of the <image> tags beginning, i.e. the location of the token representing '<'
            pixel_values (Tensor | None):    shape (b N T c h w). Optional.
            visual_features (Tensor | None):         shape (b N q d). Optional.
                If pixel_values already have been passed through encode_resample_visuals(), 
                you can pass the resampled visual embeddings via this parameter.
                If provided, pixel_values will be ignored
            head_mask (Tensor | None): TODO
            inputs_embeds (Tensor | None): TODO
            use_cache (bool): whether to return the inner keys and values. Used to speed up text generation at inference. defaults to False
            past_key_values (tuple): tuple of past_key_values of (1) the xattn layers (2) the language model
            return_dict (bool): Whether to return a dictionary. Right now, only dicts are supported, so this must be set to True. Defaults to True.
            labels (Tensor): 
                It is possible to pass the exact value as input_ids also as labels. If present, the output will contain a CE loss of the next token prediction.
                optional, defaults to None
            **kwargs

        Returns:
            (CausalLMOutputWithPast): an object containing all the useful stuff. Refer to hf documentation.

        """

        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"
        assert (input_ids is None) != (inputs_embeds is None), "you must pass either input_ids or inputs_embeds!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        xattn_past_key_values = None if past_key_values is None else past_key_values[0]
        lm_past_key_values = None if past_key_values is None else past_key_values[1]
        
        if visual_features is None:
            if xattn_past_key_values is None and pixel_values is not None:
                # extract from pixels
                assert pixel_values.size(0) == batch_size, \
                    "pixel_values must have the same batch size as the textual input!"
                
                visual_features = self.encode_resample_visuals(pixel_values)
                
            else:
                # we don't need visual_features is past is defined.
                # use dummy values, since are only required for the shape
                # visual_embedings shape (b N q d)
                visual_features = torch.zeros(
                    (batch_size, 1, self.config.resampler_num_latents, self.config.dim_visual),
                    dtype=torch.float32,
                    device=device
                )

        if media_locations is None:
            media_locations = torch.zeros(size=(batch_size, seq_length), dtype=torch.int, device=device)

        # condition xattn layers
        for i, xattn in enumerate(self.get_modified_layers()):
            layer_past = None if xattn_past_key_values is None else xattn_past_key_values[i]
            xattn.condition(visual_features, media_locations, layer_past)

        # pass through LM
        out: BaseModelOutputWithPast = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            return_dict=True,
            **kwargs
        )

        logits: torch.Tensor = self.lm_head(out.last_hidden_state)

        # collect the past_key_values from the xattn layers
        if use_cache:
            xattn_past_key_values = []
            for modified_layer in self.get_modified_layers():
                xattn_past_key_values.append(modified_layer.kv_output)

        loss = None
        if labels is not None:
            # loss function calculation, inspired by hf implementations
            # Shift so that tokens < n predict n
            # logits shape (batch, seq_length, #words)
            shift_logits = logits[..., :-1, :].contiguous()
            # labels shape (batch, seq_length)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction=loss_reduction)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(tuple(xattn_past_key_values), out.past_key_values) if use_cache else None,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
