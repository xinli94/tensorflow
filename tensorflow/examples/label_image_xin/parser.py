import pandas as pd

with open('/home/xli/tensorflow/bazel-bin/tensorflow/examples/siemens/out2', 'rb') as f:
	lines = f.readlines()

records = []
for idx in range(0, len(lines), 7):
	file = lines[idx].strip()
	ground_truth = file.split('/')[-2].strip()

	result, score = lines[idx+2].split(']')[-1].split(':')
	result = result[:-3].strip()
	score = float(score.strip())

	is_correct = ground_truth == result
	records.append([file, result, score, ground_truth, is_correct])
	# print [file, result, score, ground_truth, is_correct]

df = pd.DataFrame.from_records(records, columns=['file', 'result', 'score', 'ground_truth', 'is_correct'])
df.to_csv('parsed_out', index=False)
