# from datasets import load_dataset
from collections import defaultdict

# # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

# datasets = ["qasper"]

# for dataset in datasets:
#     data = load_dataset('THUDM/LongBench', dataset, split='test')

# context_to_queries = defaultdict(list)

# for sample in data:
#     context = sample["context"]
#     query = sample["input"]
#     context_to_queries[context].append(query)

# # Print stats
# print(f"Unique contexts: {len(context_to_queries)}")

# # Example: show one context and its queries
# for ctx, queries in list(context_to_queries.items()):
#     print("\nContext length:", len(ctx))
#     print("Number of queries for this context:", len(queries))
#     # print("Queries:")
#     # for q in queries:
#     #     print("-", q)

from datasets import load_dataset, disable_caching

# datasets = ["coursera", "gsm100", "quality", "topic_retrieval_longchat", "tpo", "codeU", "sci_fi" ,"financial_qa", "gov_report_summ", "legal_contract_qa", "meeting_summ", "multidoc_qa", "narrative_qa", "natural_question", "news_summ", "paper_assistant", "patent_summ", "review_summ", "scientific_qa", "tv_show_summ"]
# datasets = ["coursera"]
# # The corresponding NAMEs in the paper
# # "coursera", "GSM(16-shot)", "QuALITY", "TopicRet", "TOFEL", "codeU", "SFiction", "LongFQA", "GovReport", "CUAD", "QMSum", "MultiDoc2Dial", "NarrativeQA", "NQ", "Multi-news", "Openreview", "BigPatent", "SPACE", "Qasper", "SummScreen"]


# for testset in datasets:
#     # disable_caching()  uncomment this if you cannot download codeU and sci_fi
#     data = load_dataset('L4NLP/LEval', testset, split='test')
#     # evaluate your model

context_to_queries = defaultdict(list)

# for sample in data:
#     print(sample.keys())
#     context = sample["input"]
#     query = sample["instructions"]
#     print(len(query))
#     context_to_queries[context].append(query)

# # Print stats
#

# # Example: show one context and its queries


from datasets import load_dataset

# DATASET_NAMES = [
#     "hotpotwikiqa_mixup", "loogle_SD_mixup", "loogle_CR_mixup", "loogle_MIR_mixup", \
#     "multifieldqa_en_mixup", "multifieldqa_zh_mixup", "factrecall_en", "factrecall_zh", \
#     "cmrc_mixup", "lic_mixup", "dureader_mixup"
# ]

DATASET_NAMES = ["hotpotwikiqa_mixup"]
DATASET_LENGTH_LEVEL = ["16k"]


def get_dataset_names(dataset_names, length_levels):
    datasets = []
    for name in dataset_names:
        for length in length_levels:
            datasets.append(f"{name}_{length}")
    return datasets


for dataset in get_dataset_names(DATASET_NAMES, DATASET_LENGTH_LEVEL):
    data = load_dataset("Infinigence/LVEval", dataset, split="test", token=True)

for sample in data:
    context = sample["context"]
    query = sample["input"]
    context_to_queries[context].append(query)

print(f"Unique contexts: {len(context_to_queries)}")

for ctx, queries in list(context_to_queries.items()):
    print("\nContext length:", len(ctx))
    print("Number of queries for this context:", len(queries))
    # print("Queries:")
    # for q in queries:
    #     print("-", q)
