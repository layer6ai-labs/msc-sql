import os
import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import concurrent.futures
from sql_metadata import Parser as SQLParser

class BirdEvaluator:
    def __init__(self):
        self.exec_result = []
    
    def result_callback(self, result):
        self.exec_result.append(result.result())

    def execute_sql(self, sql, db_path):
        """
            Connects to a db and executes sql
        """
        # Connect to the database
        conn = sqlite3.connect(db_path)
        # Create a cursor object
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return results

    def execute_model(self, sql, db_place, difficulty, idx):
        """
            Connects to a db and executes sql and handles errors
            Returns: 
                {
                    'sql_idx': idx,
                    'results': ...
                }
        """
        try:
            result = func_timeout(30.0, self.execute_sql, args=(sql, db_place))
        except KeyboardInterrupt:
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout',)]
        except Exception as e:
            # print('exception: {}'.format(e))
            result = [(f'error', e)]  # possibly len(query) > 512 or not executable

        result = {'sql_idx': idx, 'results': result, 'sql': sql, 'difficulty': difficulty}
        return result

    def run_sqls_parallel(self, sqls_with_path, num_cpus=1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = []
            for i, sql_item in enumerate(sqls_with_path):
                future = executor.submit(self.execute_model, sql_item['sql'], sql_item['db_path'], sql_item['difficulty'], i)
                future.add_done_callback(self.result_callback)
                futures.append(future)    
    
    def sort_results(self, list_of_dicts):
        return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

    def compute_execution_accuracy(self, gt_results, pred_results):
        """
            Computes %age of prediction results matching gt results.
        """
        num_correct = 0
        num_queries = len(gt_results)
        result_list = []

        result_breakdown_by_difficulty = {}

        for i, result in enumerate(gt_results):

            # Update the totals by difficulty
            if result['difficulty'] not in result_breakdown_by_difficulty:
                result_breakdown_by_difficulty[result['difficulty']] = {'correct': 0, 'total': 1}
            else:
                result_breakdown_by_difficulty[result['difficulty']]['total'] += 1

            if set(result['results']) == set(pred_results[i]['results']):
                num_correct += 1
                result_list.append(100.0)
                result_breakdown_by_difficulty[result['difficulty']]['correct'] += 1
            else:
                result_list.append(0.0)

        acc = (num_correct / num_queries) * 100


        # Accuracy by difficulty:
        for difficulty in result_breakdown_by_difficulty:
            result_breakdown_by_difficulty[difficulty]['accuracy'] = result_breakdown_by_difficulty[difficulty]['correct']/ result_breakdown_by_difficulty[difficulty]['total'] * 100
            # remove the other keys 
            # result_breakdown_by_difficulty[difficulty].pop('correct')
            # result_breakdown_by_difficulty[difficulty].pop('total')

        return acc, result_list, result_breakdown_by_difficulty

    def evaluate(self, preds, gts, analyze=True):
        """
            expects preds, gts to be a list of dicts with 'sql' and 'db_path' keys
            and preds[i], gts[i] refers to the same ith questions SQL statement.

            If analyze is True, returns a dictionary with counts of syntax, table, column and result errors.
            
            returns accuracy on the list of predictions as well as
            returns a list of individual results corresponding to the same order as preds and gts were passed.
        """

        assert len(preds) == len(gts), "preds and gts len not equal during eval"
        self.exec_result = []
        self.run_sqls_parallel(preds, num_cpus=32)

        # results are sorted in the same order as preds
        pred_results = self.sort_results(self.exec_result)

        self.exec_result = []
        self.run_sqls_parallel(gts, num_cpus=32)

        # results are sorted in the same order as gts
        gt_results = self.sort_results(self.exec_result)

        exec_accuracy, result_list, accuracy_by_difficulty = self.compute_execution_accuracy(gt_results=gt_results, pred_results=pred_results)
        
        analysis = self.analyze_sql_outputs(gt_results, pred_results) if analyze else {}

        return {'exec': exec_accuracy, 'individual_results': result_list, 'accuracy_by_difficulty': accuracy_by_difficulty, 'analysis': analysis}


    def evaluate_single(self, pred, gt, analyze=False):
        pred_result = self.execute_model(pred['sql'], pred['db_path'], pred['difficulty'], pred['idx'])
        gt_result = self.execute_model(gt['sql'], gt['db_path'], gt['difficulty'], gt['idx'])
        exec_accuracy, _, _ = self.compute_execution_accuracy(gt_results=[gt_result], pred_results=[pred_result])
        analysis = self.analyze_sql_outputs([gt_result], [pred_result]) if analyze else {}

        return {'exec': exec_accuracy, 'analysis': analysis}


    def analyze_sql_outputs(self, gt_results, pred_results):
        result_counts = {
            'syntax_correct': 0,
            'tables_match': 0,
            'columns_match': 0,
            'num_timeouts': 0
        }

        for gt_result, pred_result in zip(gt_results, pred_results):
            gt_sql = gt_result['sql']
            pred_sql = pred_result['sql']

            if len(pred_result['results']) > 0 and pred_result['results'][0] == 'timeout':
                result_counts['num_timeouts'] += 1

            # Syntax check
            try:
                SQLParser(pred_sql)
                # The SQLParser doesn't catch all incorrect syntax, so we also check for 'error' in the results
                if len(pred_result['results']) > 0 and pred_result['results'][0] == 'error' and 'syntax error' in pred_result['results'][1]:
                    result_counts['syntax_error'] += 1
                else:
                    result_counts['syntax_correct'] += 1
            except Exception as e:
                pass

            # Table check
            try:
                pred_tables = SQLParser(pred_sql).tables
            except Exception as e:
                pred_tables = []

            try:
                gt_tables = SQLParser(gt_sql).tables
            except Exception as e:
                gt_tables = []
                print("Error in parsing gt sql tables: ", gt_sql)
                
            if set(map(str.lower, gt_tables)) == set(map(str.lower, pred_tables)):
                result_counts['tables_match'] += 1

            # Column check
            try:
                gt_columns = SQLParser(gt_sql).columns
            except Exception as e:
                gt_columns = []
                print("Error in parsing gt sql columns: ", gt_sql)

            try: 
                pred_columns = SQLParser(pred_sql).columns
            except Exception as e:
                pred_columns = []

            if set(map(str.lower, gt_columns)) == set(map(str.lower, pred_columns)):
                result_counts['columns_match'] += 1

        percent_results = {
            "syntax_valid_percent": (result_counts['syntax_correct'] / len(pred_results)) * 100,
            "tables_match_percent": (result_counts['tables_match'] / len(pred_results)) * 100,
            "columns_match_percent": (result_counts['columns_match'] / len(pred_results)) * 100,
            "num_timeouts": result_counts['num_timeouts']
        }

        return percent_results
