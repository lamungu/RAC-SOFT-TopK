import os
import json
import numpy as np
import torch
import yaml
from tqdm import tqdm
import jsonlines

from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from models.scores import *
from make_data.preprocessing import preprocess_tweet, preprocess_tweets


class Inference:
    def __init__(self, retriever, classifier, scorer, tokenizer, configs) -> None:
        self.retriever = retriever
        self.classifier = classifier
        self.scorer = scorer
        self.tokenizer = tokenizer

        self.configs = configs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever.to(self.device)
        self.classifier.to(self.device)

    @torch.no_grad()
    def infer_retriever(self, data, to_save=True):

        if to_save:
            os.makedirs(
                os.path.dirname(self.configs.output_file_retriever), exist_ok=True
            )
            output_file = open(self.configs.output_file_retriever, "w")
        data_retrieve_tweets_with_scores = []
        for data_sample in data:
            retrieve_tweets_with_scores = []
            rumor = data_sample["rumor"]
            rumor = preprocess_tweet(rumor, language=self.configs.language)
            rumor_embed = self.retriever(
                **self.tokenizer(
                    rumor,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.configs.max_length,
                ).to(self.device)
            )
            timeline_ids = []
            scores_list = []
            with tqdm(
                total=len(data_sample["timeline"]),
                desc=f"Evaluation: ",
                unit="tweet",
            ) as pbar:
                for i in range(0, len(data_sample["timeline"]), 64):
                    timeline_tweets = []
                    for j in range(i, min(i + 64, len(data_sample["timeline"]))):
                        timeline = data_sample["timeline"][j]
                        timeline_ids.append(timeline[1])
                        timeline_tweets.append([timeline[2]])
                        pbar.update(1)
                    timeline_tweets = preprocess_tweets(
                        timeline_tweets, language=self.configs.language
                    )
                    timeline_tweets = [tweet[0] for tweet in timeline_tweets]
                    timeline_embed = self.retriever(
                        **self.tokenizer(
                            timeline_tweets,
                            return_tensors="pt",
                            padding="max_length",
                            truncation=True,
                            max_length=self.configs.max_length,
                        ).to(self.device)
                    )
                    query_repr = rumor_embed["last_hidden_state"]
                    doc_repr = timeline_embed["last_hidden_state"].unsqueeze(0)
                    # print(query_repr.shape, doc_repr.shape)
                    scores = self.scorer(query_repr, doc_repr)
                    # print(scores.shape)
                    scores_list.append(scores)

                scores_list = torch.cat(scores_list, dim=1)
                scores_list = scores_list.cpu()
                # get top k scores and their corresponding tweet ids
                if len(data_sample["timeline"]) < self.configs.top_k:
                    top_k_scores = scores_list.squeeze().tolist()
                    top_k_indices = list(range(len(data_sample["timeline"])))
                else:
                    top_k_scores, top_k_indices = torch.topk(
                        scores_list, self.configs.top_k
                    )
                    top_k_scores = top_k_scores.numpy().squeeze().tolist()
                    top_k_indices = top_k_indices.numpy().squeeze().tolist()
                top_k_tweet_ids = []
                for idx in top_k_indices:
                    top_k_tweet_ids.append(timeline_ids[idx])
                if to_save:
                    for i, (idx, score) in enumerate(
                        zip(top_k_tweet_ids, top_k_scores)
                    ):
                        output_file.write(
                            f"{data_sample['id']}\tQ0\t{idx}\t{i+1}\t{score}\t{self.configs.exp_name}\n"
                        )
                # print(data_sample["timeline"][0])
                for idx, score in zip(top_k_indices, top_k_scores):
                    temp = data_sample["timeline"][idx]
                    temp.append(score)
                    retrieve_tweets_with_scores.append(temp)
            label = data_sample.pop("label", None)
            if label:
                data_retrieve_tweets_with_scores.append(
                    {
                        "id": data_sample["id"],
                        "claim": data_sample["rumor"],
                        "label": label,
                        "predicted_evidence": retrieve_tweets_with_scores,
                    }
                )
            else:
                data_retrieve_tweets_with_scores.append(
                    {
                        "id": data_sample["id"],
                        "claim": data_sample["rumor"],
                        "predicted_evidence": retrieve_tweets_with_scores,
                    }
                )
        if to_save:
            output_file.close()

        return data_retrieve_tweets_with_scores

    @torch.no_grad()
    def infer_classifier(self, data, to_save=True):
        meta_info = json.load(open(self.configs.meta_file))["ids_2_labels"]
        retrieved_data = self.infer_retriever(data, to_save=to_save)
        sep_token = self.tokenizer.sep_token
        classified_data = []
        for data_sample in retrieved_data:
            query = preprocess_tweet(
                data_sample["claim"], language=self.configs.language
            )
            res = []
            tot_score = 0
            for doc_repo in data_sample["predicted_evidence"]:
                doc = preprocess_tweet(doc_repo[2], language=self.configs.language)
                query_doc = query + sep_token + doc
                query_doc = self.tokenizer(
                    query_doc,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.configs.max_length,
                ).to(self.device)
                outputs = self.classifier(**query_doc)["logits"].cpu().squeeze()
                res.append(doc_repo[-1] * outputs.numpy())
                tot_score += doc_repo[-1]
            res = np.array(res) / tot_score
            res = res.mean(axis=0)
            res = np.argmax(res)
            classified_data.append(
                {
                    "id": data_sample["id"],
                    "predicted_label": meta_info[str(res)].upper(),
                    "claim": data_sample["claim"],
                    "label": data_sample["label"],
                    "predicted_evidence": data_sample["predicted_evidence"],
                }
            )
        os.makedirs(os.path.dirname(self.configs.output_file_classifier), exist_ok=True)
        json.dump(
            classified_data,
            open(self.configs.output_file_classifier, "w"),
            ensure_ascii=False,
        )

    @torch.no_grad()
    def infer_classifier_2(self, data, to_save=True):
        meta_info = json.load(open(self.configs.meta_file))["ids_2_labels"]
        retrieved_data = self.infer_retriever(data, to_save=to_save)
        sep_token = self.tokenizer.sep_token
        classified_data = []
        for data_sample in retrieved_data:
            query = preprocess_tweet(
                data_sample["claim"], language=self.configs.language
            )
            res = []
            tot_score = 0
            for doc_repo in data_sample["predicted_evidence"]:
                doc = preprocess_tweet(doc_repo[2], language=self.configs.language)
                query_doc = query + sep_token + doc
                query_doc = self.tokenizer(
                    query_doc,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.configs.max_length,
                ).to(self.device)
                outputs = self.classifier(**query_doc)["logits"].cpu().squeeze()
                res.append(doc_repo[-1] * outputs.numpy())
                tot_score += doc_repo[-1]
            res = np.array(res) / tot_score
            res = res.mean(axis=0)
            res = np.argmax(res)
            classified_data.append(
                {
                    "id": data_sample["id"],
                    "predicted_label": meta_info[str(res)].upper(),
                    # "claim": data_sample["claim"],
                    # "label": data_sample["label"],
                    "predicted_evidence": data_sample["predicted_evidence"],
                }
            )
        os.makedirs(os.path.dirname(self.configs.output_file_classifier), exist_ok=True)
        # json.dump(
        #     classified_data,
        #     open(self.configs.output_file_classifier, "w"),
        #     ensure_ascii=False,
        # )
        with jsonlines.open(self.configs.output_file_classifier, "w") as writer:
            for item in classified_data:
                writer.write(item)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    args = argparse.Namespace(**configs)
    tokenizer = BertTokenizer.from_pretrained(configs["model_name"])
    retriever_path = (
        configs["retriever_path"]
        if configs["retriever_path"] is not None
        else configs["model_name"]
    )
    retriever = BertModel.from_pretrained(retriever_path)
    classifier_path = (
        configs["classifier_path"]
        if configs["classifier_path"] is not None
        else configs["model_name"]
    )
    classifier = BertForSequenceClassification.from_pretrained(
        classifier_path, num_labels=configs["num_labels"]
    )
    scorer = CosineScores()

    data = json.load(open(configs["data_file"]))

    inference = Inference(retriever, classifier, scorer, tokenizer, args)
    inference.infer_classifier(data)
