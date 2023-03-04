"""
This model packs the encoder and the decoder into a seq2seq model with the help of
the EncoderDecoderModel in transformers. The forward pass is different from the
original implementation, as the encoder returns entity embeddings apart from the
text token embeddings.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import torch
from torch.nn import CrossEntropyLoss
from transformers import EncoderDecoderModel
from transformers.models.encoder_decoder.modeling_encoder_decoder import \
    shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput

from t5_lmgnn import DragonEncoderOutput


@dataclass
class T5DragonOutput(Seq2SeqLMOutput):
    link_losses: Union[List, tuple] = None


class T5Seq2seq(EncoderDecoderModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Copied from EncoderDecoderModel.forward v4.26.1
        The only difference is that we return the link losses from the encoder
        """

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # At generation time, input_ids are created from the decoder_start_token_id, encoder_outputs are
        # prepared beforehand with the encoder
        if decoder_input_ids is None and decoder_inputs_embeds is None and encoder_outputs is not None:
            decoder_input_ids = input_ids

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = DragonEncoderOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
    
        return T5DragonOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            link_losses=encoder_outputs.link_losses,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        token_type_ids=None,
        output_mask=None,
        node_ids=None,
        node_type_ids=None,
        node_scores=None,
        adj_lengths=None,
        special_nodes_mask=None,
        edge_index=None,
        edge_type=None,
        pos_triples=None,
        neg_nodes=None,
        input_embeds=None,
        cache_output=None,
        decoder_input_ids=None,
        **kwargs):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
            "output_mask": output_mask,
            "node_ids": node_ids,
            "node_type_ids": node_type_ids,
            "node_scores": node_scores,
            "adj_lengths": adj_lengths,
            "special_nodes_mask": special_nodes_mask,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "pos_triples": pos_triples,
            "neg_nodes": neg_nodes,
            "input_embeds": input_embeds,
            "cache_output": cache_output,
            "decoder_input_ids": decoder_input_ids,
            # **kwargs,
        }
