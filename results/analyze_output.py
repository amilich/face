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

"""
(defaultdict(<class 'float'>, {'Bill_Clinton': (0.40155949901180427, 6), 'George_W_Bush': (0.7987652364221773, 154), 'Hamid_Karzai': (0.5481186151590313, 6), 'John_Negroponte': (0.6990670763802291, 8), 'Michael_Bloomberg': (0.4643102593955913, 3), 'Serena_Williams': (0.8724178450044021, 15), 'Tony_Blair': (0.731153231417943, 35), 'Vladimir_Putin': (0.7158375565435591, 10)}), defaultdict(<class 'float'>, {'Bill_Clinton': 0.75, 'George_W_Bush': 0.9746835443037974, 'Hamid_Karzai': 1.0, 'John_Negroponte': 1.0, 'Michael_Bloomberg': 0.6, 'Serena_Williams': 1.0, 'Tony_Blair': 0.8333333333333334, 'Vladimir_Putin': 0.7692307692307693}))

(defaultdict(<class 'float'>, {'Bill_Clinton': (0.39714295983873066, 3), 'George_W_Bush': (0.78280167300289, 155), 'Hamid_Karzai': (0.5857328092138017, 3), 'John_Negroponte': (0.2609480149321244, 2), 'Serena_Williams': (0.7454604500721475, 12), 'Tony_Blair': (0.4454889481691731, 25), 'Vladimir_Putin': (0.5587294580845706, 5)}), defaultdict(<class 'float'>, {'Bill_Clinton': 0.375, 'George_W_Bush': 0.9810126582278481, 'Hamid_Karzai': 0.5, 'John_Negroponte': 0.25, 'Michael_Bloomberg': 5.0, 'Serena_Williams': 0.8, 'Tony_Blair': 0.5952380952380952, 'Vladimir_Putin': 0.38461538461538464}))
For Bill_Clinton, dif=-0.40155949901180427
For George_W_Bush, dif=-0.7987652364221773
For Hamid_Karzai, dif=-0.5481186151590313
For John_Negroponte, dif=-0.6990670763802291
For Michael_Bloomberg, dif=-0.4643102593955913
For Serena_Williams, dif=-0.8724178450044021
For Tony_Blair, dif=-0.731153231417943
For Vladimir_Putin, dif=-0.7158375565435591

cls=Bill_Clinton raw=0.75 noisy=0.375 num=6
cls=George_W_Bush raw=0.9746835443037974 noisy=0.9810126582278481 num=154
cls=Hamid_Karzai raw=1.0 noisy=0.5 num=6
cls=John_Negroponte raw=1.0 noisy=0.25 num=8
cls=Michael_Bloomberg raw=0.6 noisy=5.0 num=3
cls=Serena_Williams raw=1.0 noisy=0.8 num=15
cls=Tony_Blair raw=0.8333333333333334 noisy=0.5952380952380952 num=35
cls=Vladimir_Putin raw=0.7692307692307693 noisy=0.38461538461538464 num=10
[Finished in 0.6s]
"""

"""
(defaultdict(<class 'float'>, {'Bill_Clinton': (0.40155949901180427, 6), 'George_W_Bush': (0.7987652364221773, 154), 'Hamid_Karzai': (0.5481186151590313, 6), 'John_Negroponte': (0.6990670763802291, 8), 'Michael_Bloomberg': (0.4643102593955913, 3), 'Serena_Williams': (0.8724178450044021, 15), 'Tony_Blair': (0.731153231417943, 35), 'Vladimir_Putin': (0.7158375565435591, 10)}), defaultdict(<class 'float'>, {'Bill_Clinton': 0.75, 'George_W_Bush': 0.9746835443037974, 'Hamid_Karzai': 1.0, 'John_Negroponte': 1.0, 'Michael_Bloomberg': 0.6, 'Serena_Williams': 1.0, 'Tony_Blair': 0.8333333333333334, 'Vladimir_Putin': 0.7692307692307693}))

(defaultdict(<class 'float'>, {'Bill_Clinton': (0.10366666666666667, 3), 'George_W_Bush': (0.5651073825503358, 149), 'Hamid_Karzai': (0.39879999999999993, 5), 'John_Negroponte': (0.2215, 2), 'Serena_Williams': (0.5427000000000001, 10), 'Tony_Blair': (0.229625, 24), 'Vladimir_Putin': (0.1945, 4)}), defaultdict(<class 'float'>, {'Bill_Clinton': 0.375, 'George_W_Bush': 0.9430379746835443, 'Hamid_Karzai': 0.8333333333333334, 'John_Negroponte': 0.25, 'Michael_Bloomberg': 5.0, 'Serena_Williams': 0.6666666666666666, 'Tony_Blair': 0.5714285714285714, 'Vladimir_Putin': 0.3076923076923077}))
For Bill_Clinton, dif=-0.40155949901180427
For George_W_Bush, dif=-0.7987652364221773
For Hamid_Karzai, dif=-0.5481186151590313
For John_Negroponte, dif=-0.6990670763802291
For Michael_Bloomberg, dif=-0.4643102593955913
For Serena_Williams, dif=-0.8724178450044021
For Tony_Blair, dif=-0.731153231417943
For Vladimir_Putin, dif=-0.7158375565435591

cls=Bill_Clinton raw=0.75 noisy=0.375 num=6
cls=George_W_Bush raw=0.9746835443037974 noisy=0.9430379746835443 num=154
cls=Hamid_Karzai raw=1.0 noisy=0.8333333333333334 num=6
cls=John_Negroponte raw=1.0 noisy=0.25 num=8
cls=Michael_Bloomberg raw=0.6 noisy=5.0 num=3
cls=Serena_Williams raw=1.0 noisy=0.6666666666666666 num=15
cls=Tony_Blair raw=0.8333333333333334 noisy=0.5714285714285714 num=35
cls=Vladimir_Putin raw=0.7692307692307693 noisy=0.3076923076923077 num=10
[Finished in 0.3s]
"""