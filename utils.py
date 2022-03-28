def typed_entity_marker(text, i_start, i_end, i_type, j_start, j_end, j_type):
    TYPE = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
    if i_start < j_start:
      new_text = text[:i_start] + '@' + '*' + TYPE[i_type] '*' + text[i_start:i_end + 1] + '@' + \
                 text[i_end + 1:j_start] + '#' + '^' + TYPE[j_type] + '^' + text[j_start:j_end + 1] + '#' + text[j_end + 1:]
    else:
      new_text = text[:j_start] + '@' + '*' + TYPE[j_type] '*' + text[j_start:j_end + 1] + '@' + \
                 text[j_end + 1:i_start] + '#' + '^' + TYPE[i_type] + '^' + text[i_start:i_end + 1] + '#' + text[i_end + 1:]

def entity_marker(text, i_start, i_end, i_type, j_start, j_end, j_type):
    if i_start < j_start:
        new_text = text[:i_start] + '[' + i_type + ']' + text[i_start:i_end + 1] + '[/' + i_type + ']' \
                text[i_end + 1:j_start] + '[' + j_type + ']' + text[j_start:j_end + 1] + '[/' + j_type + ']' + text[j_end + 1:]
    else:
        new_text = text[:j_start] + '[' + j_type + ']' + text[j_start:j_end + 1] + '[/' + j_type + ']' + \
                 text[j_end + 1:i_start] + '[' + i_type + ']' + text[i_start:i_end + 1] + '[/' + i_type ']' + text[i_end + 1:]
