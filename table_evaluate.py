
def table_selection_evaluation(results):
    pred_tables_list = []
    gt_tables_list = []
    for result in results:
        gt_tables_list.append(result['tables_gt'].split(', '))
        pred_tables_list.append(result['tables_pred_union'])

    total_samples = len(pred_tables_list)
    total_accuracy = 0
    filtered_accuracy = 0
    total_precision = 0
    total_recall = 0

    for pred_tables, reference_tables in zip(pred_tables_list, gt_tables_list):
        
        # Convert to lowercase and strip whitespace for comparison
        predicted_tables = [x.lower().replace("--","").replace("**","").strip() for x in pred_tables]
        reference_tables = [x.lower().strip() for x in reference_tables]
        
        # Calculate accuracy
        if set(predicted_tables) == set(reference_tables):
            total_accuracy += 1
        
        # Calculate precision and recall
        true_positives = len(set(predicted_tables) & set(reference_tables))
        false_positives = len(set(predicted_tables) - set(reference_tables))
        false_negatives = len(set(reference_tables) - set(predicted_tables))

        if true_positives == len(reference_tables):
            filtered_accuracy += 1
        
        if len(predicted_tables) > 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
        
        # print(predicted_tables, reference_tables)
        # print("Precision:", precision)
        # print("Recall:", recall)
        total_precision += precision
        total_recall += recall

    # Calculate average precision and recall
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples

    # Calculate total accuracy
    accuracy = total_accuracy / total_samples
    filtered_accuracy = filtered_accuracy / total_samples

    print("Total Accuracy:", accuracy)
    print("Filtered Accuracy:", filtered_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)

    results = {
        "accuracy": accuracy,
        "filtered_accuracy": filtered_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
    }
    return results

import json
results = json.load(open("test_stage1/final_results.json", 'r'))
# for result in results:
#     json_results.append(json.loads(result.strip()))
#     print(json_results)
#     exit()
# from utils import load_json_file
# json_results = load_json_file("test_stage1/final_results.json")
table_selection_evaluation(results)
