import collections

def read_into_dicts(filename):
	cls_confidence = collections.defaultdict(float)
	cls_counts = collections.defaultdict(int)
	cls_pct_cor = collections.defaultdict(float)
	with open(filename, 'r') as raw_output:
		for line in raw_output.readlines():
			dat = line.split(' ')
			pred_cls = dat[0].split('=')[1]
			actual_cls = dat[1].split('=')[1]
			confidence = dat[2].split('=')[1]
			if actual_cls == pred_cls:
				cls_counts[actual_cls] += 1
				cls_confidence[actual_cls] += float(confidence)
			cls_pct_cor[actual_cls] += 1

		for k,v in cls_confidence.items():
			cls_confidence[k] = (v / cls_counts[k], cls_counts[k])
			cls_pct_cor[k] = cls_counts[k] / cls_pct_cor[k]

	return (cls_confidence, cls_pct_cor)

def main():
	(raw_cls_confidence, raw_cls_pct_cor) = raw_dat = read_into_dicts('raw_output')
	(noise_cls_confidence, noise_cls_pct_cor) = noise_dat = read_into_dicts('sp_noise_output')
	print(raw_dat)
	print()
	print(noise_dat)

	for k,v in raw_cls_confidence.items():
		if k not in noise_dat:
			print('For {}, dif={}'.format(k, -v[0]))
		else:
			conf_dif = v[0] - noise_cls_confidence[k][0]
			print('For {}, dif={}'.format(k, conf_dif))
	print()
	for k,v in raw_cls_pct_cor.items():
		print('cls={} raw={} noisy={} num={}'.format(k, v, noise_cls_pct_cor[k], raw_cls_confidence[k][1]))

if __name__ == '__main__':
	main()
