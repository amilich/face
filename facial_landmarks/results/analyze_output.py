import collections

def read_into_dicts(filename):
	tp = collections.defaultdict(float)
	tn = collections.defaultdict(float)
	fp = collections.defaultdict(float)
	fn = collections.defaultdict(float)
	num_ex = collections.defaultdict(int)

	classes = set()
	with open(filename, 'r') as raw_output:
		for line in raw_output.readlines():
			dat = line.split(' ')
			pred_cls = dat[0].split('=')[1]
			actual_cls = dat[1].split('=')[1]
			classes.add(actual_cls)

	with open(filename, 'r') as raw_output:
		for line in raw_output.readlines():
			dat = line.split(' ')
			pred_cls = dat[0].split('=')[1]
			actual_cls = dat[1].split('=')[1]
			confidence = dat[2].split('=')[1]
			if actual_cls == pred_cls:
				tp[actual_cls] += 1
			else:
				fp[pred_cls] += 1
				fn[actual_cls] += 1
			for cls_name in classes:
				if cls_name != actual_cls and cls_name != pred_cls:
					tn[cls_name] += 1
			num_ex[actual_cls] += 1

	return (tp, tn, fp, fn, num_ex)

def calc_cls_dat(data):
	(tp_dat, tn_dat, fp_dat, fn_dat, num_ex) = data
	cls_acc = {}
	cls_prec = {}
	cls_recall = {}
	for cls_name,tp in tp_dat.items():
		tn, fp, fn = tn_dat[cls_name], fp_dat[cls_name], fn_dat[cls_name]
		accuracy = (tp + tn) / (tp + tn + fp + fn)
		precision = (tp) / (tp + fp)
		recall = (tp) / (tp + fn)
		cls_acc[cls_name] = accuracy
		cls_prec[cls_name] = precision
		cls_recall[cls_name] = recall
	return (cls_acc, cls_prec, cls_recall, num_ex)

def print_data(data, data_std):
	(cls_acc, cls_prec, cls_recall, num_ex) = data
	(std_acc, std_prec, std_recall, _) = data_std
	acc_dict = {}
	for cls_name,accuracy in cls_acc.items():
		precision = cls_prec[cls_name]
		recall = cls_recall[cls_name]

		acc_std = std_acc[cls_name]
		prec_std = std_acc[cls_name]
		recall_std = std_recall[cls_name]

		ac_diff = accuracy - acc_std
		prec_diff = precision - prec_std
		recall_diff = recall - recall_std

		acc_dict[cls_name] = accuracy

		p_str = '{:17s} {:1.3f} {:1.3f} {:1.3f} {:3d}'.format(\
					cls_name,
					accuracy,\
					precision,\
					recall,\
					num_ex[cls_name],\
					width=15)
	return acc_dict

def main():
	with open('adv_train_out') as f:
	# with open('adv_crop_1') as f:
		cls_dat = collections.defaultdict(int)
		cls_dat_c = collections.defaultdict(int)
		for line in f.readlines():
			dat = line.split(' ')
			dat = line.split(' ')
			pred_cls = dat[0].split('=')[1]
			actual_cls = dat[1].split('=')[1]
			confidence = dat[2].split('=')[1]
			if pred_cls == actual_cls:
				cls_dat_c[actual_cls] += 1
			cls_dat[actual_cls] += 1

	for k,v in cls_dat.items():
		print(k)
		print(cls_dat_c[k] / cls_dat[k])
	quit()
	raw_dat = calc_cls_dat(read_into_dicts('raw_output'))
	noise_dat = calc_cls_dat(read_into_dicts('sp_noise_output'))
	adv_dat = calc_cls_dat(read_into_dicts('adv_train_out'))

	adv_dat_1 = calc_cls_dat(read_into_dicts('adv_crop_1'))
	a_dicts = []
	print('Raw')
	raw_d = print_data(raw_dat, raw_dat)
	print('\nNoisy')
	noise_d = print_data(noise_dat, raw_dat)
	print('\nAdv train')
	adv_train_d = print_data(adv_dat, raw_dat)
	print('\nActual adv')
	adv_landmark = print_data(adv_dat_1, raw_dat)
	
	overall_d = collections.defaultdict(list)
	def add_to_d(a_dict):
		for k,v in a_dict.items():
			overall_d[k].append(v)
	add_to_d(raw_d)
	add_to_d(noise_d)
	add_to_d(adv_train_d)
	add_to_d(adv_landmark)

	max_len = 0
	for k,v in overall_d.items():
		if len(v) > max_len:
			max_len = len(v)
	print('max len = {}'.format(max_len))
	print('\t\t\t\t  Raw Noisy Adv_train Adv_landmark')
	for k,v in overall_d.items():
		if len(v) == max_len:
			p_str = '{:17s} '.format(k)
			for i in v:
				p_str += '{:1.3f} '.format(i, width=15)
			print(p_str)

if __name__ == '__main__':
	main()
