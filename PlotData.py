import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'figure.max_open_warning': 0})
from sklearn.metrics import roc_curve, auc

def ROCCurve(X_Test, y_Test, model, ptitle =""):
	print("Generating ROC Curve...")
	fig = plt.figure()
	fpr, tpr, _ = roc_curve(y_Test, model.predict(X_Test))
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
	plt.title("ROC Curve: %s" % ptitle)
	plt.legend(loc="lower right")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	print("ROC Curve generated.")
	return fig

def PlotPerformance(history, ptitle = ""):
	print("Generating performance plot...")
	range_epochs = range(0, len(history[history.keys()[0]]))
	fig = plt.figure()
	plt.title("performance: %s" % ptitle)
	plt.xlabel("epoch")
	plt.ylabel("loss/accuracy")
	plt.ylim(0, 1)
	for res in history.keys():
		plt.plot(range_epochs, history[res], label=res)
	plt.legend(loc='upper right')
	print("Performance plot generated. ")
	return fig

def PlotData(dataframes, feature = None, weights = None, mis_feat = -999, datavtest = True):
	fig = plt.figure()
	if datavtest:
		sig = dataframes[0][feature]
		bac = dataframes[1][feature]
		sigl = (sig != mis_feat).index.tolist()
		bacl = (bac != mis_feat).index.tolist()
		plt.hist(sig[sigl], weights = weights[0][sigl], bins=100, histtype="step", color="red", label="signal", stacked=True)
		plt.hist(bac[bacl], weights = weights[1][bacl], bins=100, histtype="step", color="blue", label="background", stacked=True)
	else:
		plt.hist(dataframes[0],bins=100,histtype="step", color="red", label="signal",stacked=True)
		plt.hist(dataframes[1],bins=100,histtype="step", color="blue", label="background",stacked=True)

	plt.legend(loc='upper right')
	plt.title(feature)
	return fig

def SavePlots(figs, filename="dataplot.pdf"):
	with PdfPages(filename) as pdf:
		for fig in figs:
			pdf.savefig(fig)
	print("Plots were saved as: %s" % filename)
