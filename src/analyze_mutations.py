import sys
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from lifelines import KaplanMeierFitter
import seaborn as sns
from lifelines.statistics import logrank_test

def main():
	sns.set_style("whitegrid")
	study:str = 'nsclc_mskcc_2018'
	sample_file = f"../data/{study}/data_clinical_sample.txt"
	patient_file = f"../data/{study}/data_clinical_patient.txt"
	mutation_file = f"../data/{study}/data_mutations.txt"

	mutation_info =pd.read_csv(mutation_file,sep="\t")
	
	sample_info = pd.read_csv(sample_file,sep="\t",skiprows=4)
	sample_info = sample_info[['SAMPLE_ID', 'PATIENT_ID','TMB_NONSYNONYMOUS']]

	patient_info = pd.read_csv(patient_file,sep="\t",skiprows=4)

	
	patient_info = patient_info[['PATIENT_ID','PFS_MONTHS',
		'PFS_STATUS','PDL1_EXP','NONSYNONYMOUS_MUTATION_BURDEN',
		'PREDICTED_NEOANTIGEN_BURDEN','BEST_OVERALL_RESPONSE']]
	patient_info['PFS_STATUS'] = patient_info['PFS_STATUS'].apply(lambda x: int(x.split(":")[0]))
	# sub = patient_info[['PATIENT_ID','NONSYNONYMOUS_MUTATION_BURDEN']]


	# print(patient_info['BEST_OVERALL_RESPONSE'].value_counts())

	patient_info = patient_info.merge(sample_info,on='PATIENT_ID')
	
	patient_info['Cut1'] = pd.qcut(patient_info['NONSYNONYMOUS_MUTATION_BURDEN'],labels=['Q1','Q2','Q3','Q4'],q=4,retbins=False)
	patient_info['Cut1'] = patient_info['Cut1'].apply(lambda x: "Upper Quartile" if x=="Q4" else "Lower Three")
	
	patient_info['Cut2'] = pd.qcut(patient_info['PREDICTED_NEOANTIGEN_BURDEN'],labels=['Q1','Q2','Q3','Q4'],q=4,retbins=False)
	patient_info['Cut2'] = patient_info['Cut2'].apply(lambda x: "Upper Quartile" if x=="Q4" else "Lower Three")
	# sub['Cut2'] = pd.qcut(sub['TMB_NONSYNONYMOUS'],labels=['Q1','Q2','Q3','Q4'],q=4,retbins=False)
	# sub['Cut2'] = sub['Cut2'].apply(lambda x: "Upper Quartile" if x=="Q4" else "Lower Three")
	# print((sub['Cut1']==sub['Cut2']).all())
	kmf = KaplanMeierFitter()

	ax = plt.subplot(111)
	times = {}
	events ={}
	for group in ['Upper Quartile', 'Lower Three']:
		temp = patient_info[patient_info['Cut1']==group]
		times[group] = temp['PFS_MONTHS'].values
		events[group] = temp['PFS_STATUS'].values
		kmf.fit(temp['PFS_MONTHS'],temp['PFS_STATUS'],label = group)
		ax = kmf.plot_survival_function(ax=ax,ci_show=False)
	
	results = logrank_test(times['Upper Quartile'], 
			times['Lower Three'], 
			event_observed_A=events['Upper Quartile'], 
			event_observed_B=events['Lower Three'])

	# results.print_summary()
	# print(results.p_value)      
	# print(results.test_statistic) # 0.0872
	print(f"TMB - \tLog-rank p-value: {np.round(results.p_value,4)}")
	ax.set(title = f"Kaplan Meier Curves for TMB Stratification",xlabel = 'Time (Months)',ylabel= "Survival Probability")
	plt.savefig("../figs/TMB_Upper_vs_Lower_Third.png",dpi=500)
	plt.close()
	ax = plt.subplot(111)
	times = {}
	events ={}
	for group in ['Upper Quartile', 'Lower Three']:
		temp = patient_info[patient_info['Cut2']==group]
		times[group] = temp['PFS_MONTHS'].values
		events[group] = temp['PFS_STATUS'].values
		kmf.fit(temp['PFS_MONTHS'],temp['PFS_STATUS'],label = group)
		ax = kmf.plot_survival_function(ax=ax,ci_show=False)
	
	results = logrank_test(times['Upper Quartile'], 
			times['Lower Three'], 
			event_observed_A=events['Upper Quartile'], 
			event_observed_B=events['Lower Three'])

	# results.print_summary()
	# print(results.p_value)      
	# print(results.test_statistic) # 0.0872
	print(f"Neoantigen - \tLog-rank p-value: {np.round(results.p_value,3)}")
	ax.set(title = f"Kaplan Meier Curves for Neoantigen Load Stratification",xlabel = 'Time (Months)',ylabel= "Survival Probability")
	plt.savefig("../figs/Neoantigen_Upper_vs_Lower_Third.png",dpi=500)
	plt.close()

	patient_info = patient_info[patient_info['BEST_OVERALL_RESPONSE']!='NE']
	patient_info['Binary Response'] = patient_info['BEST_OVERALL_RESPONSE'].apply(lambda x: "Responder" if x in ['PR','CR'] else "Non-Responder")
	print(patient_info['Binary Response'].value_counts())
	ax = plt.subplot(111)
	times = {}
	events ={}
	for group in ['Responder', 'Non-Responder']:
		temp = patient_info[patient_info['Binary Response']==group]
		times[group] = temp['PFS_MONTHS'].values
		events[group] = temp['PFS_STATUS'].values
		kmf.fit(temp['PFS_MONTHS'],temp['PFS_STATUS'],label = group)
		ax = kmf.plot_survival_function(ax=ax,ci_show=False)
	
	results = logrank_test(times['Responder'], 
			times['Non-Responder'], 
			event_observed_A=events['Responder'], 
			event_observed_B=events['Non-Responder'])
	# print("responders")
	# results.print_summary()
	print(f"resp vs non-resp Log-rank p-value: {np.round(results.p_value,8)}")
	ax.set(title = f"Kaplan Meier Curves for Responders and Non-Responders",xlabel = 'Time (Months)',ylabel= "Survival Probability")
	plt.savefig("../figs/R_vs_NR.png",dpi=500)
	plt.close()

	resp = patient_info[['SAMPLE_ID','Binary Response']]
	resp.columns = ['Tumor_Sample_Barcode','Response']
	# print(resp)
	mut_ = mutation_info[['Tumor_Sample_Barcode','Hugo_Symbol','Variant_Classification','tumor_vaf']]
	mut_ = mut_.merge(resp,on='Tumor_Sample_Barcode')
	mut_['Variant_Classification'] = mut_['Variant_Classification'].apply(lambda x: " ".join(x.split("_")))
	# mut_= mut_[mut_['tumor_vaf']>=0.1]
	mut_ = mut_[mut_['Variant_Classification']!='IGR']
	for vaf_cut in [0.1, 0.2, 0.3]:
		temp = mut_[mut_['tumor_vaf']>=vaf_cut]
		R = temp[temp['Response']=='Responder']
		NR = temp[temp['Response']=='Non-Responder']
		RC = R['Variant_Classification'].value_counts()
		NC = NR['Variant_Classification'].value_counts()
		RC_var = list(RC[:5].index)
		NC_var = list(NC[:5].index)
		
		all_vars = sorted(list(set(RC_var).union(set(NC_var))))
		
		RC = pd.DataFrame(RC)
		RC.reset_index(drop=False,inplace=True)
		RC['Response']= 'Responder'
		# print(RC)
		RC['count'] = RC['count']/np.sum(RC['count'].values)
		NC = pd.DataFrame(NC)
		NC.reset_index(drop=False,inplace=True)
		NC['Response']= 'Non-Responder'
		NC['count'] = NC['count']/np.sum(NC['count'].values)
		df = pd.concat([RC,NC])
		df = df[df['Variant_Classification'].isin(all_vars)]
		df.to_csv(f"../results/variant_{vaf_cut}.csv")
		ax = sns.barplot(df,x='Variant_Classification',y='count',hue='Response')
		ax.set(ylabel = "Count of Mutation Instances",xlabel = "Variant Classification", title = f"Counts of Variant Classifications\n VAF Cutoff at {vaf_cut}")
		plt.savefig(f"../figs/VAF_{vaf_cut}.png",dpi=500)
		plt.close()

	for vaf_cut in [0.1, 0.2, 0.3]:
		temp = mut_[(mut_['tumor_vaf']>=vaf_cut) & (mut_['Variant_Classification']=='Missense Mutation')]
		R = temp[temp['Response']=='Responder']
		NR = temp[temp['Response']=='Non-Responder']
		RC = R['Hugo_Symbol'].value_counts()
		NC = NR['Hugo_Symbol'].value_counts()
		
		print("--------")
		print(vaf_cut)
		print(np.sum(NC))
		print(np.sum(RC))
		print("--------")
		RC = (RC/np.sum(RC))*100
		NC = (NC/np.sum(NC))*100




		RC_var = list(RC[:5].index)
		NC_var = list(NC[:5].index)
		
		all_vars = sorted(list(set(RC_var).union(set(NC_var))))
		
		RC = pd.DataFrame(RC)
		RC.reset_index(drop=False,inplace=True)
		RC['Response']= 'Responder'
		
		NC = pd.DataFrame(NC)
		NC.reset_index(drop=False,inplace=True)
		NC['Response']= 'Non-Responder'

		df = pd.concat([RC,NC])
		df = df[df['Hugo_Symbol'].isin(all_vars)]
		df.to_csv(f"../results/genes_{vaf_cut}.csv")
		ax = sns.barplot(df,x='Hugo_Symbol',y='count',hue='Response')
		ax.set(ylabel = "Percentage of Missense Mutation Instances",xlabel = "Gene", title = f"Percent of Mutation Instances per Gene\n VAF Cutoff at {vaf_cut}")
		plt.savefig(f"../figs/MUT_{vaf_cut}.png",dpi=500)
		plt.close()
		
	
	# print(mut_['Variant_Classification'].value_counts())
	# ax = sns.barplot(data = mut_[mut_['Variant_Classification']=='Missense_Mutation'], x='tumor_vaf',hue='Response')
	# plt.show()



if __name__ == '__main__':
	main()