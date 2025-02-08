
from metrics.bird.evaluator import BirdEvaluator

def format_results_html(results, column_names, max_rows=5, max_cols=5, max_col_width=20):
    """
        Formats the SQL results into a nicely formatted HTML table with full column headers,
        truncated column values, and no row numbers.
    """
    # Truncate rows and columns
    truncated_results = results[:max_rows]
    truncated_results = [row[:max_cols] for row in truncated_results]
    truncated_columns = column_names[:max_cols]
    
    # Determine the width for each column
    col_widths = [max(len(col), max_col_width) for col in truncated_columns]
    
    # Format each value to fit within its respective column width
    def format_value(value, width):
        return str(value)[:width]
    
    # Create HTML table
    formatted_output = "<table border='1' cellpadding='5' cellspacing='0'>\n"
    
    # Create header row
    formatted_output += "  <tr>\n"
    for col, width in zip(truncated_columns, col_widths):
        formatted_output += f"    <th>{format_value(col, width)}</th>\n"
    formatted_output += "  </tr>\n"
    
    # Create data rows
    for row in truncated_results:
        formatted_output += "  <tr>\n"
        for value, width in zip(row, col_widths):
            formatted_output += f"    <td>{format_value(value, width)}</td>\n"
        formatted_output += "  </tr>\n"
    
    formatted_output += "</table>"
    
    return formatted_output


def format_results_markdown(results, column_names, max_rows=5, max_cols=5, max_col_width=20):
    """
        Formats the SQL results into a nicely formatted table-like string with full column headers,
        truncated column values, and row numbers removed.
    """
    # Truncate rows and columns
    truncated_results = results[:max_rows]
    truncated_results = [row[:max_cols] for row in truncated_results]
    truncated_columns = column_names[:max_cols]
    
    # Determine the width for each column
    col_widths = [max(len(col), max_col_width) for col in truncated_columns]
    
    # Format each value to fit within its respective column width
    def format_value(value, width):
        return str(value)[:width].ljust(width)
    
    # Create formatted output
    formatted_output = ""
    
    # Create separator lines
    separator = "+".join(["-" * (width + 2) for width in col_widths])
    
    # Create header row
    header = "| " + " | ".join(format_value(col, width) for col, width in zip(truncated_columns, col_widths)) + " |"
    formatted_output += f"+{separator}+\n{header}\n+{separator}+\n"
    
    # Create data rows
    for row in truncated_results:
        formatted_row = "| " + " | ".join(format_value(value, width) for value, width in zip(row, col_widths)) + " |"
        formatted_output += f"{formatted_row}\n"
    
    # Add final separator
    formatted_output += f"+{separator}+"
    
    return formatted_output

def process_results_entry(entry, method='html'):

    if method == 'html':
        format_results = format_results_html
    elif method == 'markdown':
        format_results = format_results_markdown
    else:
        raise ValueError(f"Invalid method: {method}")

    bird_eval = BirdEvaluator()
        
    sql = entry['sql_pred']
    idx = entry['idx']
    db_path = entry['db_path']

    sql_result, col_names = bird_eval.execute_model_with_cols(sql, db_path, difficulty=None, idx=idx)

    if sql_result == ['timeout']:
        entry['result'] = "The query timed out"
    elif len(sql_result) == 2 and sql_result[0] == 'sql_error':
        entry['result'] = "SQL Error Occurred"
    else:
        entry['result'] = format_results(sql_result, col_names)
