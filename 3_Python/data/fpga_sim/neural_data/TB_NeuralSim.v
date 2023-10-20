`timescale 1ns / 1ps
// -----------------------------------------------------------------
// Company: UDE-ES
// Design Name: Testbench for running dataset simulator
// Generate file on: 09/15/2023, 10:31:04
// Target Devices: Simulation file
// -----------------------------------------------------------------

module TB_NeuralSim();
	localparam CLK_CYC = 15'd25000;

	reg clk, nrst, en_sim;
	wire signed[11:0] data_out0;
	wire sim_done0;

	// Choice if files with loading from memory (*_Mem) or from module (*_Mod)
	NeuralSim_Mem DUT0(
		.CLK_ADC(clk),
		.nRST(nrst),
		.EN(en_sim),
		.DATA_OUT(data_out0),
		.DATA_END(sim_done0)
	);

	// Control scheme for getting the data dependent on the sampling clock
	always begin
		#(CLK_CYC) clk = ~clk;
	end

	initial begin
		nrst = 1'd0;
		clk = 1'd0;
		en_sim = 1'd0;

		# (6* CLK_CYC);   nrst <= 1'b1;
		# (6* CLK_CYC);   nrst <= 1'b0;
		# (6* CLK_CYC);   nrst <= 1'b1;
		en_sim = 1'b1;
	end

endmodule