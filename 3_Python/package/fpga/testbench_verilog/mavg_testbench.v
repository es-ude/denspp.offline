`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 24.10.2024 23:32:06
// Design Name: 
// Module Name: TB_FIR
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module TB_MAVG();
    localparam CLK_SYS = 15'd5;
    localparam BITSIZE = 6'd16, LENGTH=6'd5;
    localparam CLK_CYC = 2* LENGTH * CLK_SYS;

    reg clk_sys, clk_adc, nrst, en_dut;
    reg [BITSIZE-'d1:0] filter_in;
    wire [BITSIZE-'d1:0] dout;
	reg [BITSIZE-'d1:0] filter_out;
	wire filter_rdy;
	
    // --- Control scheme for clk and rst
	always begin
		#(CLK_SYS) clk_sys = ~clk_sys;
	end	
	
	// --- Control scheme for catching data
	always@(posedge filter_rdy) begin
		filter_out = dout;
	end	
	
	// --- Using DUT
	Filter_MAVG_{$device_id}#(BITSIZE, LENGTH, 1'd1, 'd31) DUT(
	   .CLK(clk_sys),
	   .nRST(nrst),
	   .EN(en_dut),
	   .START_FLAG(clk_adc),
	   .DATA_IN(filter_in),
	   .DATA_OUT(dout),
	   .DATA_VALID(filter_rdy)
	);

    integer ite;
    initial begin
		nrst = 1'd1;
		clk_sys = 1'd0;
		clk_adc = 1'd0;
		en_dut = 1'd0;
		
		filter_in = ('d1 << BITSIZE -'d1) + 'd100;
		filter_out = ('d1 << BITSIZE -'d1);
		
        
        // Step #1: Reset
        #(3* CLK_SYS);  nrst <= 1'd0;
		#(6* CLK_SYS);  nrst <= 1'b1;
		#(6* CLK_SYS);  nrst <= 1'b0;
		#(6* CLK_SYS);  nrst <= 1'b1;
		
		// Step #2: Activate DUT
		#(4* CLK_SYS);  en_dut <= 1'd1;
		#(8* CLK_SYS);
		
		// Step #3: Run filter
		for(ite='d0; ite < 2*LENGTH; ite=ite+'d1) begin
		    //Apply data and run filter
		    #(2* CLK_CYC - 2 * CLK_SYS)     clk_adc = 1'd1;
                                            filter_in = filter_in;  
            #(2* CLK_SYS);                  clk_adc = 1'd0;
		end
		#(2* CLK_CYC);
		$stop;
	end
endmodule
