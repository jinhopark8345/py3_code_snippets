d_struct = {'train': 'train', 'test': 'test'}

mapped = [{'features': ['id', 'tokens', 'bboxes', 'ner_tags', 'image'], 'num_rows': 149},
          {'features': ['id', 'tokens', 'bboxes', 'ner_tags', 'image'], 'num_rows': 50}]


# tmp =  dict(zip(d_struct.keys(), mapped))
tmp =  dict(zip(['train', 'test'], mapped))
breakpoint()
