def entity_marker1(text, i_start, i_end, i_type,j_start, j_end, j_type):
    new_text = ''
    TYPE = {'ORG':'단체','PER':'사람','DAT':'날짜','LOC':'위치','POH':'기타','NOH':'수량'}
    if i_start < j_start:
        new_text = text[:i_start] + '*' + text[i_start:i_end + 1] + '[' + TYPE[i_type] + ']*' + text[i_end + 1:j_start] + '*' + \
                text[j_start:j_end + 1] + '[' + TYPE[j_type] + ']*'  + text[j_end + 1:]
    else:
        new_text = text[:j_start] + '*' + text[j_start:j_end + 1] + '[' + TYPE[j_type] + ']*' + text[j_end + 1:i_start] + '*' + \
                text[i_start:i_end + 1] + '[' + TYPE[i_type] + ']*' + text[i_end + 1:]
    return new_text

def entity_marker2(text, i_start, i_end, i_type, j_start, j_end, j_type):
    new_text = ''
    TYPE = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
    if i_start < j_start:
      new_text = text[:i_start] + '@' + '*' + TYPE[i_type] + '*' + text[i_start:i_end + 1] + '@' + \
                text[i_end + 1:j_start] + '#' + '^' + TYPE[j_type] + '^' + text[j_start:j_end + 1] + '#' + text[j_end + 1:]
    else:
      new_text = text[:j_start] + '#' + '^' + TYPE[j_type] + '^' + text[j_start:j_end + 1] + '#' + \
                text[j_end + 1:i_start] + '@' + '*' + TYPE[i_type] + '*' + text[i_start:i_end + 1] + '@' + text[i_end + 1:]
    return new_text

def entity_marker3(text, i_start, i_end, i_type, j_start, j_end, j_type):
    new_text = ''
    TYPE = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
    if i_start < j_start:
      new_text = text[:i_start] + '@' + '*' + TYPE[i_type] + '*' + '[E1]' + text[i_start:i_end + 1] + '[/E1]' + '@' + \
                text[i_end + 1:j_start] + '#' + '^' + TYPE[j_type] + '^' + '[E2]' + text[j_start:j_end + 1] + '[/E2]' + '#' + text[j_end + 1:]
    else:
      new_text = text[:j_start] + '#' + '^' + TYPE[j_type] + '^' + '[E2]' + text[j_start:j_end + 1] + '[/E2]' + '#' + \
                text[j_end + 1:i_start] + '@' + '*' + TYPE[i_type] + '*' + '[E1]' + text[i_start:i_end + 1] + '[/E1]' + '@' + text[i_end + 1:]
    return new_text

def entity_marker4(text, i_start, i_end, i_type, j_start, j_end, j_type):
    new_text = ''
    TYPE = {'ORG':'단체', 'PER':'사람', 'DAT':'날짜', 'LOC':'위치', 'POH':'기타', 'NOH':'수량'}
    if i_start < j_start:
      new_text = text[:i_start] + '[E1]' + '@' + '*' + TYPE[i_type] + '*' + text[i_start:i_end + 1] + '@' + '[/E1]' + \
                text[i_end + 1:j_start] + '[E2]' + '#' + '^' + TYPE[j_type] + '^' + text[j_start:j_end + 1] + '#' + '[/E2]' + text[j_end + 1:]
    else:
      new_text = text[:j_start] + '[E2]' + '#' + '^' + TYPE[j_type] + '^' + text[j_start:j_end + 1] + '#' + '[/E2]' + \
                text[j_end + 1:i_start] + '[E1]' + '@' + '*' + TYPE[i_type] + '*' + text[i_start:i_end + 1] + '@' + '[/E1]' + text[i_end + 1:]
    return new_text


