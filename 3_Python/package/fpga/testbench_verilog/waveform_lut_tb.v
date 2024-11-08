`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:             University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:            AE
// Design Name:         Testbench for Digital Direct Syntheziser
// Generate file on:    10/30/2024, 12:52:20
// Copied on: 	        {$date_copy_created}
// Target Devices:      Simulation file
// Comments:            Multply CNT_VAL_WAIT with (1 + 1/(size(sine_lutram)-1)) for right timing
//////////////////////////////////////////////////////////////////////////////////


module TB_LUT_WVF();
	localparam CLK_CYC_NS = 'd5, CNT_VAL_WAIT = 'd{$wait_cycles};
	localparam CNT_PERIODS = 'd{$period_cycles}, NUM_PERIODS = 'd4;

	// --- CODE FOR READING DATA FROM EXTERNAL
    {$do_read_lut_external}wire {$signed_type} [{$num_sinelut} * {$bitsize_lut} - 'd1:0] LUT_ROM;
    {$do_read_lut_external}assign LUT_ROM = {{$lut_data_stream}};

    // --- Control lines
	reg CLK_SYS, nRST, EN_LUT, TRGG_LUT;
	wire {$signed_type} [{$bitsize_lut}-'d1:0] LUT_DATA;
    wire END_LUT;

	LUT_WVF_GEN{$device_id} DUT(
		.CLK_SYS(CLK_SYS),
		.nRST(nRST),
		.EN(EN_LUT),
		{$do_cnt_external}.WAIT_CYC(CNT_VAL_WAIT),
		{$do_read_lut_external}.LUT_ROM(LUT_ROM),
		{$do_trgg_external}.TRGG_CNT_FLAG(TRGG_LUT),
		{$do_cnt_external}.WAIT_CYC(CNT_VAL_WAIT),
		.LUT_VALUE(LUT_DATA),
		.LUT_END(END_LUT)
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
