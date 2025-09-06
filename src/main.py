import json
import asyncio
import argparse
import pytrec_eval
from algorithm import realm, format_result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Realm."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="dl19",
        help="Dataset name (used to locate ../data/retrieve_results_{dataset}.json and qrels_{dataset}.json)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="google/flan-t5-large",
        choices=["google/flan-t5-large","google/flan-t5-xl","google/flan-t5-xxl","gpt-5"],
        help="Which LLM to use."
    )
    parser.add_argument(
        "--order", "-o",
        type=str,
        default="bm25",
        choices=["bm25","random","inverse"],
        help="Initial order before reranking."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset
    input_path='../data/retrieve_results_'+dataset+'.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qrels_path='../data/qrels_'+dataset+'.json'
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    for key in qrels:
        for doc in qrels[key]:
            qrels[key][doc]=int(qrels[key][doc])
    for i in range(len(data)):
        docs = data[i]['hits']
        if len(docs)==0:
            continue
        id=str(docs[0]['qid'])
        n_items = len(docs)
        for j in range(n_items):
            if docs[j]['docid'] in qrels[id]:
                data[i]['hits'][j]['qrels']=qrels[id][docs[j]['docid']]
            else:
                data[i]['hits'][j]['qrels']=0
    result=asyncio.run(realm(data,args.model,args.order))
    run = format_result(result)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    result = evaluator.evaluate(run)
    avg_ndcg = sum(res['ndcg_cut_10'] for res in result.values()) / len(result)
    print("NDCG@10:", avg_ndcg,"\n")

if __name__ == "__main__":
    main()