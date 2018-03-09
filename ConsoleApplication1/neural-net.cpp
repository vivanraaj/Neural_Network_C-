#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
//#include "aria.h"
#include <stdio.h>
#include <conio.h>

using namespace std;

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void){return m_trainingDataFile.eof();}
	void getTopology(vector<unsigned> &topology);
	string getLine()
	{
		string s;
		getline(m_trainingDataFile, s, '\n');
		return s;
	}

	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals,stringstream &is);
	unsigned getTargetOutputs(vector<double> &targetOutputVals, stringstream &is);
	void setPointerToBeggining(void) { m_trainingDataFile.clear(); m_trainingDataFile.seekg(0, ios::beg); } // Just implementing the constructor


private:
	ifstream m_trainingDataFile;

};


double M_LeftSpeed_Max = 350;
double M_LeftSpeed_Min = 0;
double M_RightSpeed_Max = 350;
double M_RightSpeed_Min = 0;
double M_LeftDistance_Max = 5000;
double M_LeftDistance_Min = 100;
double M_FrontDistance_Max = 5000;
double M_FrontDistance_Min = 100;

double NormalizedTargets_LeftSpeed(string values) {
	double normalizedTargets_leftspeed = stod(values);
	normalizedTargets_leftspeed = (normalizedTargets_leftspeed - M_LeftSpeed_Min) / (M_LeftSpeed_Max - M_LeftSpeed_Min);
	return normalizedTargets_leftspeed;
}

double NormalizedTargets_RightSpeed(string values) {
	double normalizedTargets_rightspeed = stod(values);
	normalizedTargets_rightspeed = (normalizedTargets_rightspeed - M_RightSpeed_Min) / (M_RightSpeed_Max - M_RightSpeed_Min);
	return normalizedTargets_rightspeed;
}

double NormalizedInputs_LeftDistance(string values) {
	double normalizedInputs_leftdistance = stod(values);
	normalizedInputs_leftdistance = (normalizedInputs_leftdistance - M_LeftDistance_Min) / (M_LeftDistance_Max - M_LeftDistance_Min);
	return normalizedInputs_leftdistance;
}

double NormalizedInputs_FrontDistance(string values) {
	double normalizedInputs_frontdistance = stod(values);
	normalizedInputs_frontdistance = (normalizedInputs_frontdistance - M_FrontDistance_Min) / (M_FrontDistance_Max - M_FrontDistance_Min);
	return normalizedInputs_frontdistance;
}



void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if(this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while(!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals, stringstream &is)
{
	inputVals.clear();

	int input = 0;
	while (input<2) {
		string value;

		getline(is, value, ',');
		//cout << value << '/n';c
		double doubled_inputs;


		if (input == 0)
		{
			doubled_inputs = NormalizedInputs_LeftDistance(value);
		}
		else
		{
			doubled_inputs = NormalizedInputs_FrontDistance(value);
		}
		inputVals.push_back(doubled_inputs);

		//if (label.compare("input:") == 0) {// If we find the word "input:" then we extract the inputs:
		//double oneValue; //A variable for picking input by input
		//while (ss >> oneValue) { // While you still have inputs, then store them in the inputVals vector
		//inputVals.push_back(oneValue); // This just inserts the obtained input in the inputVals vector
		//}
		input++;
	}
	//cout << '\n' << inputVals[0] << '\n' << inputVals[1] << endl;

	return inputVals.size();
}


unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals, stringstream &is)
{
	targetOutputVals.clear();

	int outputs = 0;
	//double c, d;
	while (outputs<2) {
		string value;//A variable for picking input by input
		getline(is, value, ','); // While you still have inputs, then store them in the inputVals vector
								 //cout << value << '/n';
		double normalizedTargets;
		if (outputs == 0)
		{
			normalizedTargets = NormalizedTargets_LeftSpeed(value);
		}
		else
		{
			normalizedTargets = NormalizedTargets_RightSpeed(value);
		}
		//double normalizedTargets = stod(value);
		targetOutputVals.push_back(normalizedTargets); // This just inserts the obtained input in the inputVals vector
													   //cout << targetOutputVals.back();

		outputs++;

	}
	//cout << '\n' << targetOutputVals[0] << '\n' << targetOutputVals[1] << endl;

	return targetOutputVals.size();
}

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
public:
	// E.F. 12.12.2017 Neuron Constructor:
	Neuron(unsigned numOutputs, unsigned myIndex);
	Neuron(unsigned numOutputs, unsigned myIndex, vector<double> &loaded_weights);
	void setOutputVal(double val) { m_outputVal = val; }
	//R.V. 12.12.2017 GET OUT FUNCTION
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	//vector<Connection> m_outputWeights;
private:
	static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }  // outputs random weight between 0 and 1
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	unsigned m_myIndex;
	double m_gradient;
	vector<Connection> m_outputWeights;
	friend class Net;      // allows class Net to access its values
};

double Neuron::eta = 0.7; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = 
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
	return  1 / (1 + exp(-x));
}

double Neuron::transferFunctionDerivative(double x)
{
	return Neuron::transferFunction(x)*(1 - Neuron::transferFunction(x));
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * 
				 prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex, vector<double> &loaded_weights)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = loaded_weights.back();
		loaded_weights.pop_back();
	}

	m_myIndex = myIndex;
}



// ****************** class Net ******************
class Net
{
public:
	Net(const vector<unsigned> &topology);
	Net(const vector<unsigned> &topology, vector<double> &loaded_weights);
	void feedForward(vector<double> &inputVals);    
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) ;   
	double getRecentAverageError(void) const { return m_recentAverageError; }
	void saveToFile(string hiddenWeight);
	vector<double> loadWeights(string filepath);
	double m_recentAverageError;
	vector<Layer> m_layers;
};


void Net::getResults(vector<double> &resultVals)
{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	// Calculate overal net error (RMS of output neuron errors)

	Layer &outputLayer = m_layers.back();
	double error_1 = 0;
	double error_2 = 0;


	for (unsigned n = 0; n < outputLayer.size() - 1; ++n){

	//separate error calculation for 2 outputs
	if (n == 0)
		{
		double delta = targetVals[0] - outputLayer[0].getOutputVal();
		error_1 = delta * delta;
		error_1 = sqrt(error_1);
		error_1 /= outputLayer.size() - 1;
	}
	if (n == 1)
	{
		double delta = targetVals[1] - outputLayer[1].getOutputVal();
		error_2 = delta * delta;
		error_2 = sqrt(error_2);
		error_2 /= outputLayer.size() - 1;
	}
	// below to average the RMSE of both output
	m_recentAverageError = (error_1 + error_2) / 2;
	}


	// Calculate output layer gradients
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
	// Calculate gradients on hidden layers

	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(vector<double> &inputVals)
{

	for(unsigned i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]); 
	}

	// Forward propagate
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1];
		for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}
Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}
		m_layers.back().back().setOutputVal(1.0);
	}
}

// below we overload the constructor to take in saved weights vector
Net::Net(const vector<unsigned> &topology, vector<double> &loaded_weights)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum, loaded_weights));
		}
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::saveToFile(string hiddenWeight) {

	ofstream hiddenWeightStream;
	string hiddenPath = hiddenWeight;
	hiddenWeightStream.open(hiddenPath.c_str());

	for (int i = 0; i < m_layers.size(); i++)
	{
		for (int j = 0; j < m_layers[i].size(); j++)	// weights of 1st layer and so on
		{
			auto count = m_layers[i][j].m_outputWeights.size();
			for (int k = 0; k < count; ++k)
			{
				hiddenWeightStream << m_layers[i][j].m_outputWeights[k].weight;
				if (k != m_layers[i][j].m_outputWeights.size())

				{
					if (i != m_layers.size())
						hiddenWeightStream << ",";
				}

			}
		}
	}
	hiddenWeightStream.flush();
	hiddenWeightStream.close();
}

vector<double> Net::loadWeights(string filepath) {

	ifstream infile(filepath);
	vector<double> loaded_weights;
	string s;
	getline(infile, s);
	stringstream ss(s);

	while (!ss.eof())
	{
		string value;
		getline(ss, value, ',');
		double loaded_weights_value = stod(value);

		loaded_weights.push_back(loaded_weights_value);
	}

	reverse(loaded_weights.begin(), loaded_weights.end());
	return loaded_weights;
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}


///*
int main() {

		TrainingData trainData("C:\\Users\\mrmad\\Documents\\Essex\\OneDrive - University of Essex\\CE889-NEURALNETWORKS\\Codes Used For Demo\\Use_This_For_Assignment\\Datatrain_data_2_hidden.txt");
		TrainingData validationData("C:\\Users\\mrmad\\Documents\\Essex\\OneDrive - University of Essex\\CE889-NEURALNETWORKS\\Codes Used For Demo\\Use_This_For_Assignment\\validation_data_2_hidden.txt");

		vector<unsigned> topology;
		trainData.getTopology(topology);
		validationData.getTopology(topology);
		Net myNet(topology);

		int i = 1;

		vector<double> inputVals, targetVals, resultVals;
		vector<double> inputValuesValidationSet, targetValuesValidationSet, resultValuesValidationSet;

		while (i < 101) {     // 100 epochs
			int trainingPass = 0;
			//cout << "EPOCH NUMBER: " << i << endl;

			if (i > 1) {
				topology.clear();
				trainData.getTopology(topology);
				validationData.getTopology(topology);
			}

			while (!trainData.isEof()) {

				string l = trainData.getLine();
				stringstream is(l);
				// Get new input data and feed it forward:
				if (trainData.getNextInputs(inputVals, is) != topology[0]) {
					break;
				}

				trainData.getTargetOutputs(targetVals, is);

				//showVectorVals(": Inputs:", inputVals);
				myNet.feedForward(inputVals);

				// Collect the net's actual output results:
				myNet.getResults(resultVals);
				//showVectorVals("Outputs:", resultVals);

				// Train the net what the outputs should have been:
				//showVectorVals("Targets:", targetVals);
				assert(targetVals.size() == topology.back());

				myNet.backProp(targetVals);

				++trainingPass;
			}

			// outputs the RMSE for every epoch for the training loss
			cout << "  " << myNet.getRecentAverageError();

			///////// Validation Data Training
			
			while (!validationData.isEof()) {

				string ls = validationData.getLine();
				stringstream lstream(ls);
				// Get new input data and feed it forward:
				if (validationData.getNextInputs(inputValuesValidationSet, lstream) != topology[0]) {
					break;
				}

				validationData.getTargetOutputs(targetValuesValidationSet, lstream);

				//showVectorVals(": Inputs:", inputValuesValidationSet);
				myNet.feedForward(inputValuesValidationSet);

				// Collect the net's actual output results:
				myNet.getResults(resultValuesValidationSet);

				//showVectorVals("Outputs:", resultValuesValidationSet);

				// Train the net what the outputs should have been:
				//showVectorVals("Targets:", targetValuesValidationSet);
				//assert(targetVals.size() == topology.back());

				Layer &outputLayer = myNet.m_layers.back();
				double error_1 = 0;
				double error_2 = 0;

				for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {

					//separate error calculation for 2 outputs
					if (n == 0)
					{
						double delta = targetVals[0] - outputLayer[0].getOutputVal();
						error_1 = delta * delta;
						error_1 = sqrt(error_1);
						error_1 /= outputLayer.size() - 1;
					}
					if (n == 1)
					{
						double delta = targetVals[1] - outputLayer[1].getOutputVal();
						error_2 = delta * delta;
						error_2 = sqrt(error_2);
						error_2 /= outputLayer.size() - 1;
					}
					myNet.m_recentAverageError = (error_1 + error_2) / 2;
				}

			}
			// outputs the RMSE for every epoch for the training loss
			cout << "  " << myNet.getRecentAverageError() << endl;

			i++;

			trainData.setPointerToBeggining();
			validationData.setPointerToBeggining();
		}

		// run the save weight function
		myNet.saveToFile("final_111217.txt");
		cout << "Completed Training" << endl;
		system("pause");
		}
//*/


/*
int main(int argc, char **argv)
{
	// Initialisationss
	// create instances
	Aria::init();
	ArRobot robot;
	ArPose pose;
	// add to initialisation −> create instances
	ArSensorReading *sonarSensor[8];


	// parse command line arguments
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();
	// add to initialisation −> parse command line arguments
	argParser.addDefaultArgument("−connectLaser");

	// connect to robot (and laser, etc)
	ArRobotConnector
		robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		std::cout << "Robot connected!" << std::endl;
	// add to ”connect to ...” (after ”connect to robot”)
	ArLaserConnector laserConnector(&argParser, &robot,
		&robotConnector);
	if (laserConnector.connectLasers())
		std::cout << "Laser connected!" << std::endl;
	ArLaser *laser = robot.findLaser(1);
	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();

	//RUN BELOW FOR USING TRAINED NEURAL NETWORK
	string saveWeightPath = "weights_that_work.txt";

	vector<unsigned> topology;
	topology.push_back(2); topology.push_back(4); topology.push_back(2);

	// below load the topology and create the layers
	// read the txt file and load into a weight vector
	vector<double> loaded_weights;
	Net myNettemp(topology);
	loaded_weights = myNettemp.loadWeights(saveWeightPath);

	Net myNet(topology, loaded_weights);

	while (true)
	{
		// run
		// get sonar readings
		int sonarRange[8];
		for (int i = 0; i < 8; i++)
		{
			sonarSensor[i] = robot.getSonarReading(i);
			// since sonarSensor is a pointer
			sonarRange[i] = sonarSensor[i]->getRange();
			//std::cout << "Range before run: " + sonarRange[i];
		}


		double left_reading = sonarSensor[0]->getRange();
		cout << "left_reading: " << left_reading << std::endl;

		// denormalize 
		double Robot_Left_Input = left_reading / 5000;

		double front_reading = sonarSensor[1]->getRange();
		cout << "front_reading: " << front_reading << std::endl;

		// denormalize
		double Robot_Front_Input = front_reading / 5000;

		vector<double> inputVals, resultVals;

		inputVals.push_back(Robot_Left_Input);
		inputVals.push_back(Robot_Front_Input);

		cout << "/n" << ": Test_Inputs" << inputVals[0] << " " << inputVals[1] << endl;

		myNet.feedForward(inputVals);

		//collect the net's actual output results:
		myNet.getResults(resultVals);


		cout << ": Test_Outputs" << resultVals[0] << " " << resultVals[1] << endl;

		// denormalize
		int left_vel = resultVals[0] * 350;
		int right_vel = resultVals[1] * 350;


		cout << ": Final Velocity" << left_vel << " " << right_vel << endl;
		robot.setVel2(left_vel, right_vel);


		ArUtil::sleep(100);
	}
	// termination
	// stop the robot
	robot.lock();
	robot.stop();
	robot.unlock();
	// terminate all threads and exit

	Aria::exit();

	return 0;
}
*/