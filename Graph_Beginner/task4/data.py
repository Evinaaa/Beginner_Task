def load_data(path):
    entities, relations = set(), set()
    triples = []
    
    with open(path) as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            entities.update([h, t])
            relations.add(r)
            triples.append((h, r, t))
    
    return triples, list(entities), list(relations)