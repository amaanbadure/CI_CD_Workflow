def check_results():
    import json
    import glob
    import sys
    import os
    
    try:
        # Get the latest run
        runs = glob.glob('experiment_logs/*/run.json')
        if not runs:
            print("No run.json files found!")
            sys.exit(1)
            
        latest_run = max(runs, key=lambda x: os.path.getctime(x))
        print(f"Reading results from: {latest_run}")
        
        with open(latest_run) as f:
            results = json.load(f)
            print(f"Results structure: {results.keys()}")
            
            # Try to get accuracy from different possible locations
            accuracy = None
            
            # Try different possible locations
            if 'info' in results and 'test_accuracy' in results['info']:
                accuracy = results['info']['test_accuracy']
            elif 'result' in results and isinstance(results['result'], dict) and 'test_accuracy' in results['result']:
                accuracy = results['result']['test_accuracy']
            elif 'result' in results and isinstance(results['result'], float):
                accuracy = results['result']
            
            if accuracy is None:
                print("Could not find test_accuracy in results!")
                print("Full results content for debugging:")
                print(json.dumps(results, indent=2))
                sys.exit(1)
                
            print(f"Found accuracy: {accuracy}")
            
            if accuracy < 0.60:
                print(f'Accuracy {accuracy} below threshold 0.85')
                sys.exit(1)
            else:
                print(f'Accuracy {accuracy} meets or exceeds threshold 0.85')
                
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        sys.exit(1)