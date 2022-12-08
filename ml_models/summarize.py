import pandas as pd

#Summarize the model results

def main():
    df = pd.read_csv('model_report.csv')
    
    algorithms = ['svm', 'decision-tree', 'naive-bayes']
    representations = ['bow', 'freq', 'tfidf']
    tasks = ['classification', 'type_of_antisemitism']
    
    algo_rep_tasks = []
    
    for algorithm in algorithms:
        for representation in representations:
            algo_rep_tasks += [{'Algorithm': algorithm,
                               'Representation': representation,
                               'Task': task} for task in tasks]
    
    for algo_rep_task in algo_rep_tasks:
        algo = algo_rep_task['Algorithm']
        representation = algo_rep_task['Representation']
        task = algo_rep_task['Task']
        
        temp_df = df[df['Algorithm'] == algo]
        temp_df = temp_df[temp_df['Representation'] == representation]
        temp_df = temp_df[temp_df['Task'] == task]
        
        score = temp_df['Score'].mean()
        algo_rep_task['Score'] = score
    
    output_df = pd.DataFrame(algo_rep_tasks)
    output_df.to_csv('summarize.csv', index=False)
    
if __name__ == '__main__':
    main()
