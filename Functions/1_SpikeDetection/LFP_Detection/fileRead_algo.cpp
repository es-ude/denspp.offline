#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

using std::cout; using std::cerr;
using std::endl; using std::string;
using std::ifstream; using std::vector;

int input; short signed in;
short signed delay0 = 0; short signed delay1 = 0; short signed delay2 = 0; 
short signed coeff_a1 = 8796; float coeff_a2 = 8796; 
long signed A1; long signed A2; long signed S0; long signed S1; long signed NEO;
short signed a1; short signed a2;
short signed out;

int main()
{

    string filename("input.txt");
    vector<string> lines;
    string line;

    ifstream input_file(filename);
    if (!input_file.is_open()) {
        cerr << "Could not open the file - '"
             << filename << "'" << endl;
        return EXIT_FAILURE;
    }

    while (getline(input_file, line)){
        lines.push_back(line);
    }

    for (const auto &i : lines){
        input = stoi(i);
        in = (short)input;
        in = in/64; in = in*64;
        A1 = delay0 * coeff_a1; a1 = A1/32767;
        A2 = delay1 * coeff_a2; a2 = A2/32767;
        out = in + a1 + a2;
        S0 = out * out; S1 = out * delay0; NEO = S0 - S1;
        std::cout << NEO << " " << out << " " << a2 << std::endl ;
        delay2 = delay1;
        delay1 = delay0;
        delay0 = out;
    }


    input_file.close();
    return EXIT_SUCCESS;
}