from srtk.knowledge_graph import Wikidata as SRTKWikidata

class Wikidata(SRTKWikidata):
    """Extend Wikidata from SRTK such that it returns
    intermediate entities in two hop relations."""
    @staticmethod
    def get_id_from_uri(uri):
        """Get property id from uri."""
        return uri.split('/')[-1]

    @staticmethod
    def is_qid(qid):
        """Check if qid is a valid Wikidata entity id."""
        return qid.startswith('Q') and qid[1:].isdigit()

    @staticmethod
    def is_pid(pid):
        """Check if pid is a valid Wikidata property id."""
        return pid.startswith('P') and pid[1:].isdigit()

    def search_two_hop_relations(self, src, dst):
        """Search two hop relation between src and dst.
        It differs from the original implementation in that it returns
        not only relations but also intermediate entities.

        Args:
            src (str): source entity
            dst (str): destination entity
        
        Returns:
            list[tuple(str)]: list of paths, each path is (relation1, intermediate entity, relation2)
        """
        if not self.is_qid(src) or not self.is_qid(dst):
            return []

        query = f"""
            SELECT DISTINCT ?r1 ?x ?r2 WHERE {{
                wd:{src} ?r1 ?x.
                ?x ?r2 wd:{dst}.
                {self.get_quantifier_filter('x')}
            }}
            """
        paths = self.queryWikidata(query)
        # Keep only PIDs in the paths
        paths = [(self.get_id_from_uri(path['r1']['value']),
                  self.get_id_from_uri(path['x']['value']),
                  self.get_id_from_uri(path['r2']['value']))
                 for path in paths]
        return paths
