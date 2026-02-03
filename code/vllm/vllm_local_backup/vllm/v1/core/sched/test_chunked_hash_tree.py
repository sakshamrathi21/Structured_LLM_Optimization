import sys
sys.path.append("./build")

from chunked_hash_tree import ChunkedHashTree

tree = ChunkedHashTree(chunk_size=500)

req_id = 42   # integer, not string
tokens = [12, 94, 382, 91, 17]

tree.insert(req_id, tokens)
best, missing = tree.find_best_request()
tree.activate_request(best)
