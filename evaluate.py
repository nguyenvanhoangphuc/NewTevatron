# Tính precision, recall, map
# Cụ thể tính: precision@k và recall@k với k thuộc {1, 2, 3, 5, 7, 10}
# Tính my_recall thay vì chia cho tổng số tài liệu liên quan thì hãy chia cho giá trị nhỏ nhất của k và tổng số tài liệu liên quan
import datasets

# init
# test_ds_path = "./test_retrieval_ms_marco"
# rank_txt_path = "./beir_embedding_scifact/rank.scifact.txt"
test_ds_path = "./test_retrieval_ja"
rank_txt_path = "./beir_embedding_scifact/rank.scifact_done.txt"
ks = [1, 2, 3, 5, 7, 10]    # Danh sách các k cần tính precision và recall

# Load lại test dataset
test_ds = datasets.load_from_disk(test_ds_path)

# print(test_ds)
query_relate = {}

for i in range(len(test_ds['query_id'])): 
    # print(test_ds['query_id'][i])
    # print(test_ds['query'][i])
    # print(test_ds['positive_passages'][i])
    # print(test_ds['negative_passages'][i])
    query_relate[test_ds['query_id'][i]] = []
    for pos in test_ds['positive_passages'][i]: 
        query_relate[test_ds['query_id'][i]].append(pos['docid'])
    # break

# print(query_relate)

top10_predict = {}

# load file rank.scifact.txt
with open(rank_txt_path, "r") as f: 
    lines = f.readlines()
    for line in lines: 
        parts = line.strip().split("\t")
        query_id = parts[0]
        doc_id = parts[1]
        if query_id not in top10_predict: 
            top10_predict[query_id] = []
        top10_predict[query_id].append(doc_id)

# print(top10_predict)

# # in ra 5 query đầu tiên của query_relate và top10_predict
# count = 0
# for query_id in query_relate: 
#     print("Query id: ", query_id)
#     print("Query: ", test_ds['query'][count])
#     print("Positive passages: ", query_relate[query_id])
#     print("Top 10 predict: ", top10_predict[query_id])
#     count += 1
#     if count == 5: 
#         break

# Tính precision@k và recall@k với k thuộc {1, 3, 5, 10}

precisions = {}
recalls = {}
for k in ks: 
    precisions[k] = 0
    recalls[k] = 0

print(len(top10_predict))

for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for k in ks: 
        count_retrieval = 0
        for i in range(k): 
            if predict_passages[i] in positive_passages: 
                count_retrieval += 1
        precisions[k] += count_retrieval/k
        recalls[k] += count_retrieval/len(positive_passages)

for k in ks:
    precisions[k] /= len(top10_predict)
    recalls[k] /= len(top10_predict)

print("Precisions@k: ", precisions)
print("Recalls@k: ", recalls)

# Tính MAP
APs = []
for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    count_retrieval = 0
    sum_precision = 0
    for i in range(len(predict_passages)): 
        if predict_passages[i] in positive_passages: 
            count_retrieval += 1
            sum_precision += count_retrieval/(i+1)
    APs.append(sum_precision/len(positive_passages))

MAP = sum(APs)/len(APs)
print("MAP@10: ", MAP)

# Tính my_recall thay vì chia cho tổng số tài liệu liên quan thì hãy chia cho giá trị nhỏ nhất của k và tổng số tài liệu liên quan
my_recalls = {}
for k in ks: 
    my_recalls[k] = 0

for query_id in top10_predict:
    positive_passages = query_relate[query_id]
    predict_passages = top10_predict[query_id]
    for k in ks: 
        count_retrieval = 0
        for i in range(k): 
            if predict_passages[i] in positive_passages: 
                count_retrieval += 1
        my_recalls[k] += count_retrieval/min(k, len(positive_passages))

for k in ks:
    my_recalls[k] /= len(top10_predict)

print("My_recalls@k: ", my_recalls)