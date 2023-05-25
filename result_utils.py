def extract_score(line):
    return [float(ins.split(':')[1].strip()) for ins in line.split(',')]


def read_results(filename):
    
    with open(filename, 'r') as f:
        res = f.read()

    res = res.split('\n')
    res = [ins for ins in res if ins!='']
    
    assert len(res)%5==0
    
    results = {}

    for i in range(0, len(res), 5):
        block = res[i:i+5]
        metric = block[0].strip('*').strip()

        axes = [ins.split(':')[0].strip() for ins in block[1].split(',')]
        scores = extract_score(block[1])
        scores_known = extract_score(block[2])
        scores_unknown = extract_score(block[3])
        significance = extract_score(block[4])

        results[metric] = {'overall': scores, 'known': scores_known, 
                           'unknown': scores_unknown, 'significance': significance}
        
        
    return results


def read_results_task(task, models):
    results = {}
    for model in models:
        results[model] = read_results(f'logs/acl_submission/{task}_{model}.log')
        
    return results