`timescale 1ns / 1ps
// --------------------------------------------------------------------------------------
// Company: UDE-ES
// Design Name: Testbench for running SineWFG emulator
// Generate file on: 10/30/2024, 12:52:20
// Target Devices: Simulation file
// Comments: Multply CNT_VAL_WAIT with (1 + 1/(size(sine_lutram)-1)) for right timing
// ------------------------------------------------------------------------------------

module TB_LUT_WVF();
	localparam CLK_CYC_NS = 'd5, CNT_VAL_WAIT = 'd{$wait_cycles};
	localparam CNT_PERIODS = 'd{$period_cycles}, NUM_PERIODS = 'd4;

	reg CLK_SYS, nRST, EN_LUT;
	wire [{$bitsize_lut}-'d1:0] lut_data;
    wire lut_end;

	LUT_WVF_GEN{$device_id} DUT(
		.CLK_SYS(CLK_SYS),
		.nRST(nRST),
		.EN(EN_LUT),
		{$do_cnt_external}.WAIT_CYC(CNT_VAL_WAIT),
		{$do_read_lut_external}.LUT_ROM(),
		.LUT_VALUE(lut_data),
		.LUT_END(lut_end)
	);

	// Control scheme for getting the data dependent on the sampling clock
	always begin
		#(CLK_CYC_NS) CLK_SYS = ~CLK_SYS;
	end

	initial begin
		CLK_SYS = 1'd0;
		nRST = 1'd1;
		EN_LUT = 1'd0;

		//Step #1: Reset-Phase
		# (7* CLK_CYC_NS);   nRST <= 1'b1;
		repeat(2) begin
			# (10* CLK_CYC_NS);   nRST <= 1'b0;
			# (10* CLK_CYC_NS);   nRST <= 1'b1;
		end

		//Step #2: Enable SineLUT
		#(10* CLK_CYC_NS);   EN_LUT = 1'b1;

		//Step #3: End simulation
		#(NUM_PERIODS* CNT_PERIODS* CLK_CYC_NS);
		$stop;
	end

endmodule
