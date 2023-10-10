"""Run a REL server for entity linking. 

In my current configuration, this script must be run from the project root directory,
and in the rel environment.

Check this tutorial for more information:
- https://rel.readthedocs.io/en/latest/tutorials/e2e_entity_linking/
"""

from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
from REL.server import make_handler

BASE_URL = 'resources/rel'
WIKI_VERSION = "wiki_2019"

config = {
    "mode": "eval",
    "model_path": BASE_URL + "/ed-wiki-2019/model",  # or alias, see also tutorial 7: custom models
}

model = EntityDisambiguation(BASE_URL, WIKI_VERSION, config)

# Using Flair:
tagger_ner = load_flair_ner("ner-fast")

host = "127.0.0.1"
port = 1235
server_address = (host, port)
server = HTTPServer(
    server_address,
    make_handler(
        BASE_URL, WIKI_VERSION, model, tagger_ner
    ),
)

try:
    print(f"Ready for listening on {host}:{port} .")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
