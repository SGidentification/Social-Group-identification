import pandas as pd
import numpy as np
from collections import namedtuple
import re

Metrics = namedtuple('Metrics', ['precision','recall','f1'])

def remove_symbols(comment):
    regEx = r"[^a-zA-Z0-9,.?!'\s@#]"
    tmp_string = re.sub(regEx, "", comment) #remove symbols
    tmp_string = re.sub(r'\s+', ' ', tmp_string).strip() #remove consecutive whitespaces and whitespaces at borders
    tmp_string = re.sub(r'\s##', '##', tmp_string) #return annotation symbols to correct position
    tmp_string = re.sub(r'@@\s', '@@', tmp_string) #return annotation symbols to correct position
    return tmp_string

def convert_to_bio_format(seq):
    # Apply postprocessing to each element in the array
    bio_sequence = (len(seq))*["O"]
    inside_entity = False
    # Find the start and end indices of the gold entity
    for i, token in enumerate(seq):
        if token.startswith("@@") and token.endswith("##"):
            bio_sequence[i] = "B"
            inside_entity = False
        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1
    
            # Find the index of the next token starting with @@
            for j, t in enumerate(seq[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break
    
            # Find the index of the next token ending with ##
            for j, t in enumerate(seq[i + 1:]):
                if t.endswith("##"):
                    next_hash_index = i + 1 + j
                    break
    
            if next_at_index != -1 and next_hash_index != -1 and next_hash_index < next_at_index:
                inside_entity = True
            if next_at_index == -1 and next_hash_index != -1:
                inside_entity = True
                
        elif inside_entity:
            bio_sequence[i] = "I"
            if not token.endswith("##"):
                inside_entity = True
            else:
                inside_entity = False
    return bio_sequence



#get list of intervals of entities in bio sequence in form [start_id, end_id]
def get_entity_intervals(bio_seq):
    intervals = []
    i = 0
    while i < len(bio_seq):
        #entity found
        if bio_seq[i] == 'B':
            interval = [i,-1]
            #go over current entity
            while i+1 < len(bio_seq) and bio_seq[i+1] == 'I':
                i += 1
            interval[1] = i
            intervals.append(interval)
        i += 1
    return intervals

#f1 score on entity level (strict we only count exact matches as positives)
def f1_score_entity_level(annots1, annots2):
    #get intervals from bio sequences of the given lists of strings
    intervals1 = [get_entity_intervals(bio_seq) for bio_seq in [convert_to_bio_format(remove_symbols(annot).split()) for annot in annots1]]
    intervals2 = [get_entity_intervals(bio_seq) for bio_seq in [convert_to_bio_format(remove_symbols(annot).split()) for annot in annots2]]
    tp = 0

    #go over all documents
    for i in range(len(intervals1)):
        intervals2remove = []
        #go over all entity intervals
        for interval in intervals1[i]:
            #increase number of true positives if interval is found in both annotations
            if interval in intervals2[i]:
                tp += 1
                intervals2remove.append(interval)
        for interval in intervals2remove:
            intervals1[i].remove(interval)
            intervals2[i].remove(interval)
    
    fp = 0
    fn = 0

    #get false positives by adding number of intervals that haven't been found in annots2
    #get false negatives by adding number of intervals that haven't been found in annots1
    for i in range(len(intervals1)):
        fp += len(intervals1[i])
        fn += len(intervals2[i])

    #nothing found but there was nothing to find => precision 1
    prec = 1 if (tp + fp) == 0 else tp/(tp+fp)

    #nothing found but nothing missed => recall 1
    rec = 1 if (tp + fn) == 0 else tp/(tp+fn)

    f1 = 2*prec*rec/(prec+rec)

    return Metrics(prec, rec, f1)



#get list of id tag pairs (id, tag) without O
def get_ids_tags(bio_seq):
   return [(id, tag) for id, tag in enumerate(bio_seq) if tag != 'O']

#f1 score on token level (strict we only count exact matches as positives)
def f1_score_token_level(annots1, annots2):
    id_tag_pairs1 = [get_ids_tags(bio_seq) for bio_seq in [convert_to_bio_format(remove_symbols(annot).split()) for annot in annots1]]
    id_tag_pairs2 = [get_ids_tags(bio_seq) for bio_seq in [convert_to_bio_format(remove_symbols(annot).split()) for annot in annots2]]
    tp = 0
    
    #go over all documents
    for i in range(len(id_tag_pairs1)):
        id_tag_pairs2remove = []
        #go over all id tag pairs
        for id_tag_pair in id_tag_pairs1[i]:
            #increase number of true positives if id tag pair is found in both annotations
            if id_tag_pair in id_tag_pairs2[i]:
                tp += 1
                id_tag_pairs2remove.append(id_tag_pair)
        for id_tag_pair in id_tag_pairs2remove:
            id_tag_pairs1[i].remove(id_tag_pair)
            id_tag_pairs2[i].remove(id_tag_pair)
    
    fp = 0
    fn = 0


    for i in range(len(id_tag_pairs1)):
        #go over all id tag pairs
        ids = []
        for id_tag_pair in id_tag_pairs1[i]:
            #increment number of false positives
            fp += 1
            #store id to avoid double counting
            ids.append(id_tag_pair[0])

        #count number of false negatives, without id counted as false positives    
        fn += len([id_tag_pair for id_tag_pair in id_tag_pairs2[i] if id_tag_pair[0] not in ids])
     
    #nothing found but there was nothing to find => precision 1
    prec = 1 if (tp + fp) == 0 else tp/(tp+fp)

    #nothing found but nothing missed => recall 1
    rec = 1 if (tp + fn) == 0 else tp/(tp+fn)

    f1 = 2*prec*rec/(prec+rec)

    return Metrics(prec, rec, f1)

#mapping function for increased readability
def tag_mapping(tag):
    if tag == 'B':
        return 0
    if tag == 'I':
        return 1
    if tag == 'O':
        return 2
    raise ValueError('Invalid tag')

#calculate cohens kappa with 'O', 'O' pairs or without
def cohens_kappa(annots1, annots2, only_labeled = False):
    bio_labels1 = [convert_to_bio_format(remove_symbols(annot).split()) for annot in annots1]
    bio_labels2 = [convert_to_bio_format(remove_symbols(annot).split()) for annot in annots2]

    #calculate confusion matrix
    confusion_matrix = np.zeros((3,3))
    for comment_id in range(len(bio_labels1)):
        for token_id in range(len(bio_labels1[comment_id])):
            label1 = bio_labels1[comment_id][token_id]
            label2 = bio_labels2[comment_id][token_id]
            confusion_matrix[tag_mapping(label1)][tag_mapping(label2)] += 1
    
    #set 'O', 'O' pairs to 0 if specified
    if only_labeled:
        confusion_matrix[tag_mapping('O')][tag_mapping('O')] = 0
    
    #get observed/expected agreement
    observed_agreement = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    row_sum = np.sum(confusion_matrix, axis=0)
    column_sum = np.sum(confusion_matrix, axis=1)
    expected_agreement = np.sum(row_sum * column_sum)/(np.sum(confusion_matrix)**2)
    
    #get kappa
    kappa = (observed_agreement - expected_agreement)/(1 - expected_agreement)
    return kappa

#helper method to find errors in annotation
def check_token_number(seq1, seq2):
    for i in range(len(seq1)):
        tmp1 = remove_symbols(seq1[i])
        tmp2 = remove_symbols(seq2[i])
        if len(tmp1.split()) != len(tmp2.split()):
            print(i)
            print(f"{len(tmp1.split())}   {len(tmp2.split())}")
            print
            print(tmp1.split())
            print(tmp2.split())

#quick test for f1_score
def test(filepath):
    data = pd.read_csv(filepath, keep_default_na=False, encoding="latin-1")
    annotation1 = data['annotation2'] #change annotations here
    annotation2 = data['gold annotation'] #change annotations here
    
    metrics_token_level = f1_score_token_level(annotation1,annotation2) 
    print(f"TOKEN_LEVEL:   precision: {metrics_token_level.precision}, recall: {metrics_token_level.recall}, f1: {metrics_token_level.f1}")
    
    metrics_entity_level = f1_score_entity_level(annotation1,annotation2)
    print(f"ENTITY_LEVEL:  precision: {metrics_entity_level.precision}, recall: {metrics_entity_level.recall}, f1: {metrics_entity_level.f1}")

    print(f"kappa:              {cohens_kappa(annotation1, annotation2)}")
    print(f"kappa only labeled: {cohens_kappa(annotation1, annotation2, only_labeled=True)}")
    #check_token_number(annotation1,annotation2)




test(r"politics.csv")

