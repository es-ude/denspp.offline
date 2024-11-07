`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 13.11.2020 08:55:47
// Design Name: 
// Module Name: TB_CLK
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

module TB_MULT_LUT();
    localparam MUT_WIDTH = 6'd6, STOP = (('d1<<2*MUT_WIDTH)-'d1);
    
    wire CLK_100MHz;
    wire nRST;
    
    reg signed [MUT_WIDTH-'d1:0] DATA_A;
    reg signed [MUT_WIDTH-'d1:0] DATA_B;
    wire signed [2*MUT_WIDTH-'d1:0] MUT_REF;
    wire signed [2*MUT_WIDTH-'d1:0] MUT_OUT;
    
    reg flag_start;
    wire flag_done;
    
    assign MUT_REF = DATA_A* DATA_B;
    
    MULT_LUT_SIGNED#(MUT_WIDTH) MUT1(
        .A(DATA_A),
        .B(DATA_B),
        .Q(MUT_OUT)
    );
    
    integer i0 = 0;
    initial begin
        DATA_A <= 0;
        DATA_B <= 0;
        flag_start <= 1'd0;
        #5;
                
        //for (i0 = 0; i0 < 10; i0=i0+1) begin
        #500; flag_start <= 1'd1;   #10; flag_start <= 1'd0;
        for (i0 = 0; i0 <= STOP; i0=i0+'d1) begin
            DATA_A <= DATA_A + 1;
            if(DATA_A == ('d1<<MUT_WIDTH)-'d1) begin
                DATA_B <= DATA_B +1;
                DATA_A <= 'd0;
            end
            #500;
        end
    end    
endmodule
