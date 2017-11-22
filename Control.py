#!/usr/bin/env python2
import ProcessFile as pf
import PlotData as pld
import TrainAnalyzeModel as tam
import sys

def CompareDataPerformance(df, names, ptitle = "", ModelIndex = 0, Epochs = 10):
	print( "Run for: ", ptitle)
	Train, Test = pf.SplitData(df)
	XTrain, YTrain = pf.TrainandTarget(Train, names)
	XTest, YTest = pf.TrainandTarget(Test, names)
	p_model, p_history = tam.BuildTrainSeqModel(ModelIndex, Epochs, XTrain, YTrain, XTest, YTest)
	plots = [pld.PlotPerformance(p_history.history, ptitle + str(ModelIndex)), pld.ROCCurve(XTest, YTest, p_model, ptitle + str(ModelIndex))]
	pld.SavePlots(plots, ptitle + "MI" + str(ModelIndex) + ".pdf")

def main():
	filename=sys.argv[1]

	VarNames=["signal", "l_1_pT", "l_1_eta","l_1_phi", "l_2_pT", "l_2_eta", "l_2_phi", "MET", "MET_phi", "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1"]
	RawNames=["l_1_pT", "l_1_eta","l_1_phi", "l_2_pT", "l_2_eta", "l_2_phi"]
	FeatureNames=[ "MET", "MET_phi", "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1"]
	
	N_Max=550000
	N_Train=500000

	df = pf.LoadFile(filename, VarNames)
	print("Loading complete.")

	if RawNames[0] != 'signal':
		RawNames.insert(0, 'signal')
	if FeatureNames[0] != 'signal':
	    FeatureNames.insert(0, 'signal')
	RawFeatNames = RawNames + FeatureNames[1:]
	
	df_raw = df[RawNames]
	df_feat = df[FeatureNames]
	df_rawfeat = df[RawFeatNames]

	print("Exercise 5.2")
	CompareDataPerformance(df_raw, RawNames, "RawVariables")
	CompareDataPerformance(df_feat, FeatureNames, "FeatVariables")
	CompareDataPerformance(df_rawfeat, RawFeatNames, "RawFeatVariables")

	print("Exercise 5.3")
	CompareDataPerformance(df_rawfeat, RawFeatNames, "RawFeatVariables", 1)
	CompareDataPerformance(df_rawfeat, RawFeatNames, "RawFeatVariables", 2)
	CompareDataPerformance(df_rawfeat, RawFeatNames, "RawFeatVariables", 3)
	


	
if __name__ == "__main__":
	main()
