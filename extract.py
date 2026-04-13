import re
with open("train-seq2seq.py", "r") as f:
    lines = f.readlines()

def get_block(start_line, end_line):
    return "".join(lines[start_line-1:end_line])

def remove_blocks(blocks_to_remove):
    # blocks_to_remove is a list of (start, end)
    keep = []
    remove_set = set()
    for s, e in blocks_to_remove:
        for i in range(s, e+1):
            remove_set.add(i)
    
    for i, line in enumerate(lines):
        if (i+1) not in remove_set:
            keep.append(line)
    return "".join(keep)

# blocks to remove based on cat -n:
# 53-73 TransformerSchedule
# 615-683 SinePositionEncoding
# 684-770 verify_positional_encoding
# 771-846 build_seq2seq_model_baseline
# 847-989 build_seq2seq_model_multi_attention
# 990-1137 build_seq2seq_transformer
# 1138-1179 build_seq2seq_model (wait, let's check end of build_seq2seq_model)

