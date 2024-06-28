#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <time.h>
#include <omp.h>
#include "../eigen-3.4.0/Eigen/Eigen"
#define MAXSIZE 10000
#define BLOCKNUM 8
#ifdef BLOCKNUM
#define BLOCKSIZE (SIZE / BLOCKNUM)
#endif
#define REPEAT 1e5
#define BETA 0.85
#define EPSILON 1e-15
#define SIZE Size
//#define DEBUG
//#define DEBUGINCYCLE
#ifdef DEBUGINCYCLE
#define PRINTINCYCLE 100
#endif
using namespace std;
using namespace Eigen;

int Size = MAXSIZE;
vector<list<int>> toGraph(MAXSIZE);
vector<list<int>> fromGraph(MAXSIZE);
SparseMatrix<double> M(MAXSIZE, MAXSIZE);
VectorXd W(MAXSIZE);
VectorXd DE(MAXSIZE);
vector<VectorXd> R(2);

int readData(const string& path) {
#ifdef DEBUG
	clock_t start = clock();
#endif
	ifstream file(path);
	if (!file.is_open()) {
		cout << "Error: file not found" << endl;
		return 1;
	}
	int from, to;
	int Max = 0, Min = MAXSIZE;
	while (file >> from >> to) {
		Max = max({ Max, from, to });
		Min = min({ Min, from, to });
		from--, to--;
		toGraph[from].insert(lower_bound(toGraph[from].begin(), toGraph[from].end(), to), to);
		fromGraph[to].insert(lower_bound(fromGraph[to].begin(), fromGraph[to].end(), from), from);
	}
	toGraph.resize(Max);
	fromGraph.resize(Max);
	M.resize(Max, Max);
	W.resize(Max);
	DE.resize(Max);
	R[0].resize(Max);
	R[1].resize(Max);
	file.close();
#ifdef DEBUG
	cout << "Min = " << Min << ", Max = " << Max << endl;
	cout << "Data read";
	cout << "\tTime: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
#endif
	Size = Max;
	return 0;
}

void printGraph() {
	int count = 0;
	for (int i = 0; i < SIZE; i++) {
		if (toGraph[i].empty()  &&  fromGraph[i].empty()) continue;
		cout << i + 1 << ": ";
		for (int j : toGraph[i]) {
			cout << j + 1 << " ";
		}
		cout << endl;
		count++;
	}
	cout << "count = " << count << endl;
}

int initM() {
#ifdef DEBUG
	clock_t start = clock();
#endif
	for (int i = 0; i < SIZE; i++) {
		if (fromGraph[i].empty()) continue;
		for (int j : fromGraph[i]) {
			M.insert(i, j) = 1.0;
		}
	}
	M.makeCompressed();
#ifdef DEBUG
	cout << "M initialized";
	cout << "\tTime: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
#endif
	return 0;
}

int initW() {
#ifdef DEBUG
	clock_t start = clock();
#endif
	for (int i = 0; i < SIZE; i++) {
		W(i) = toGraph[i].empty() ? SIZE : toGraph[i].size();
	}
#ifdef DEBUG
	cout << "W initialized";
	cout << "\tTime: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
#endif
	return 0;
}

int initDE() {
#ifdef DEBUG
	clock_t start = clock();
#endif
	for (int i = 0; i < SIZE; i++) {
		DE(i) = toGraph[i].empty() ? 1.0 : 0;
	}
#ifdef DEBUG
	cout << "DE initialized";
	cout << "\tTime: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
#endif
	return 0;
}

int initR() {
#ifdef DEBUG
	clock_t start = clock();
#endif
	for (int i = 0; i < SIZE; i++) {
		R[0](i) = 1.0 / SIZE;
	}
#ifdef DEBUG
	cout << "R initialized";
	cout << "\tTime: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
#endif
	return 0;
}

int runPageRank() {
	initM();
	initW();
	initDE();
	initR();
#ifdef DEBUG
	clock_t start = clock();
#endif
	VectorXd WR;
	int cnt = 0;
	int i = 0;
	double diff = 0.0;
	double teleport_DE = 0.0;
#ifdef BLOCKNUM
	omp_set_num_threads(BLOCKNUM);
	vector<SparseMatrix<double>> BM(BLOCKNUM);
#pragma omp parallel for
	for (int j = 0; j < BLOCKNUM; j++) {
		if (j < BLOCKNUM - 1)
			BM[j] = M.block(j * BLOCKSIZE, 0, BLOCKSIZE, SIZE);
		else
			BM[j] = M.block(j * BLOCKSIZE, 0, BLOCKSIZE + SIZE % BLOCKNUM, SIZE);
	}
#endif
	while (true) {
#ifndef BLOCKNUM
		WR = R[i].cwiseQuotient(W);
		R[i ^ 1] = (BETA * ((M * WR).array() + DE.dot(WR))) + (1 - BETA) / SIZE;
#else
		WR = R[i].cwiseQuotient(W);
		teleport_DE = DE.dot(WR);
#pragma omp parallel for
		for (int j = 0; j < BLOCKNUM; j++) {
			if (j < BLOCKNUM - 1)
				R[i ^ 1].segment(j * BLOCKSIZE, BLOCKSIZE) = (BETA * ((BM[j] * WR).array() + teleport_DE)) + (1 - BETA) / SIZE;
			else
				R[i ^ 1].segment(j * BLOCKSIZE, BLOCKSIZE + SIZE % BLOCKNUM) = (BETA * ((BM[j] * WR).array() + teleport_DE)) + (1 - BETA) / SIZE;
		}
#endif
		diff = (R[i] - R[i ^ 1]).lpNorm<1>();
#ifdef DEBUGINCYCLE
		if (++cnt % PRINTINCYCLE == 0) {
			cout << "Iteration " << cnt << " done" << endl;
			cout << "diff = " << diff << endl;
			//cout << "WR = \n" << WR << endl;
			//cout << "M * WR = \n" << M * WR << endl;
			//cout << "DE * WR = \n" << DE.dot(WR) << endl;
			//cout << "R = \n" << R[i] << endl;
		}
#endif
		if (cnt > REPEAT || diff < EPSILON) break;
		i ^= 1;
	}
#ifdef DEBUG
	cout << "R = \n" << R[i] << endl;
	cout << "Time: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "Iteration " << cnt << " total" << endl;
	cout << "R.sum() = " << R[i].sum() << endl;
#endif
	return 0;
}

void printRSorted(int num) {
	vector<pair<double, int>> sorted;
	for (int i = 0; i < SIZE; i++) {
		sorted.push_back({ R[0](i), i });
	}
	sort(sorted.begin(), sorted.end(), greater<pair<double, int>>());
	for (int i = 0; i < num; i++) {
		cout << sorted[i].second + 1 << " " << sorted[i].first << endl;
	}
}

void printRSorted(int num, const string& opath) {
	ofstream file(opath);
	if (!file.is_open()) {
		cout << "Error: file not found" << endl;
		return;
	}
	vector<pair<double, int>> sorted;
	for (int i = 0; i < SIZE; i++) {
		sorted.push_back({ R[0](i), i });
	}
	sort(sorted.begin(), sorted.end(), greater<pair<double, int>>());
	for (int i = 0; i < num; i++) {
		file << sorted[i].second + 1 << " " << sorted[i].first << endl;
	}
	file.close();
}

int main() {
	clock_t start = clock();
	readData("Data.txt");
	// printGraph();
	runPageRank();
	printRSorted(100, "result.txt");
	cout << "Time: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
	return 0;
}