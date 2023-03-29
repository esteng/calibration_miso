import re 

def anonymize_level_1(sql_str):
    """
    Anonymize the SQL string by replacing literals with a generic
    """
    replace_gex = re.compile(r'\"(.*?)\"')
    sql_str = replace_gex.sub('value', sql_str)
    return sql_str 

def anonymize_level_2(sql_str):
    """
    Anonymize the SQL string by replacing literals and entities with generics
    """
    pass

def anonymize_level_3(sql_str):
    """
    Anonymize the SQL string by replacing functions, literals and entities with generics
    """
    pass