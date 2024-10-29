//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     31.01.2022 12:26:47
// Copied on: 	    {$date_copy_created}
// Module Name:     IIR Filter (1st and 2nd Order / SOS-Filter, One Cycle with five MULT)
// Target Devices:  ASIC (Implementing and using the mutArrayS module)
//                  FPGA (Using DSP block for multiplication)
// Tool Versions:   1v1
// Description:     Structure: Direct Form 2, signed integer operation
// Processing:      Data applied on posedge clk --> sampling on negedge clk
// Dependencies:    mutArrayS with custom-made multiplier
// 
// State: 	        Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    Implement the IIR filter in Direct Form 2 (less memory usage)
// Parameters:      BITWIDTH_DATA --> Bitwidth of input data
//                  uint_io --> Do type conversion (1'b1) from unsigned to signed to unsigned else don't do
//                  BITWIDTH_WEIGHTS --> Bitwidth of all weights (fraction is bitwidth)
//////////////////////////////////////////////////////////////////////////////////
{$use_ram_coeff}`define IIR_{$device_id}_USE_INT_WEIGHTS
{$use_ext_mult}`define IIR_{$device_id}_USE_EXT_MULT
{$use_lut_mult}`define IIR_{$device_id}_USE_LUT_MULT


// Input values are integer or unsigned with size of BITWIDTH_DATA (no fixed point)
// Internal operation with signed values and all weights have fraction width of BITWIDTH_DATA-'d2;
module Filter_IIR_OneCyc#(
    parameter BITWIDTH_DATA = 6'd{$bitwidth_data},
    parameter uint_io = 1'b{$signed_data},
    parameter BITWIDTH_WEIGHTS = 6'd{$bitwidth_weights}
)(
    // Global control signals
    input wire CLK,
    input wire nRST,
    input wire EN,
    input wire START_FLAG,
    // Filter coefficients input (b0, b1, b2, -a1, -a2)
	`ifndef IIR_{$device_id}_USE_INT_WEIGHTS
		input wire signed [5* BITWIDTH_WEIGHTS-'d1:0] FILT_WEIGHTS,
	`endif
	`ifdef IIR_{$device_id}_USE_EXT_MULT
        output [BITWIDTH_DATA-'d1:0] MULT_INA,
        output [BITWIDTH_WEIGHTS-'d1:0] MULT_INB,
        input [BITWIDTH_DATA+BITWIDTH_WEIGHTS-'d1:0] MULT_OUT,
    `endif
    // Data I/O
    input wire signed [BITWIDTH_DATA-'d1:0] DATA_IN,
    output wire signed [BITWIDTH_DATA-'d1:0] DATA_OUT,
    output wire DATA_VALID
);

    localparam UPPER_MASK = BITWIDTH_DATA + BITWIDTH_WEIGHTS - 'd3;

    //################## Structure of Direct Form 1 Filter ##################  
    wire signed [BITWIDTH_DATA-'d1:0] coeff [5:0];
    reg signed [BITWIDTH_DATA-'d1:0] tap_input [1:0], tap_output [1:0];
    wire signed [2*BITWIDTH_DATA-'d1:0] step_input [2:0], step_output [1:0];
    wire signed [2*BITWIDTH_DATA-'d1:0] dout [1:0], out;
    wire signed [BITWIDTH_DATA-'d1:0] data_in0;
    
    assign dout[0] = step_input[0] + step_input[1] + step_input[2];
    assign dout[1] = step_output[0] + step_output[1];
    assign out = dout[0] + dout[1];
    assign DATA_VALID = !START_FLAG && EN;
    
    assign data_in0 = {uint_io ^ DATA_IN[BITWIDTH_DATA-'d1], DATA_IN[BITWIDTH_DATA-'d2:0]};
    assign DATA_OUT = {uint_io ^ tap_output[0][BITWIDTH_DATA-'d1], tap_output[0][BITWIDTH_DATA-'d2:0]};
    
    //Control-Structure
    always@(posedge CLK) begin
        if(~(nRST && EN)) begin
            tap_input[0] <= 'd0;
            tap_input[1] <= 'd0;
            tap_output[0] <= 'd0;
            tap_output[1] <= 'd0;
        end else begin 
            tap_input[0] <= (START_FLAG) ? data_in0 : tap_input[0];
            tap_input[1] <= (START_FLAG) ? tap_input[0] : tap_input[1];
            tap_output[0] <= (START_FLAG) ? out[UPPER_MASK-:BITWIDTH_DATA] : tap_output[0];
            tap_output[1] <= (START_FLAG) ? tap_output[0] : tap_output[1];
        end
    end
    
    //################## Choicing the multiplier module ##################  
    `ifdef IIR_{$device_id}_USE_LUT_MULT   
        MULT_LUT_SIGNED#(BITWIDTH_DATA) MUT_B0(.A(data_in0),      .B(coeff_b[0]), .Q(step_input[2]));
        MULT_LUT_SIGNED#(BITWIDTH_DATA) MUT_B1(.A(tap_input[0]),  .B(coeff_b[1]), .Q(step_input[0]));
        MULT_LUT_SIGNED#(BITWIDTH_DATA) MUT_B2(.A(tap_input[1]),  .B(coeff_b[2]), .Q(step_input[1]));
        MULT_LUT_SIGNED#(BITWIDTH_DATA) MUT_A1(.A(tap_output[0]), .B(coeff_a[1]), .Q(step_output[0]));
        MULT_LUT_SIGNED#(BITWIDTH_DATA) MUT_A2(.A(tap_output[1]), .B(coeff_a[2]), .Q(step_output[1]));
    `else
        assign step_input[2] = data_in0 * coeff[0];
        assign step_input[0] = tap_input[0] * coeff[1];
        assign step_input[1] = tap_input[1] * coeff[2];
        assign step_output[0] = tap_output[0] * coeff[3];
        assign step_output[1] = tap_output[1] * coeff[4];
    `endif
    
    //################## Definition of the weights ##################
    // Filter coefficients input (b0, b1, b2, -a1, -a2), a0 is ignored due to 1
    
    `ifndef IIR_{$device_id}_USE_INT_WEIGHTS
		genvar i1;
		for(i1 = 'd0; i1 < 'd5; i1 = i1 +1) begin
			assign coeff[i1] = FILT_WEIGHTS[i1 * BITWIDTH_WEIGHTS+:BITWIDTH_WEIGHTS];
		end
	`else		
		//--- Used filter coefficients ({$filter_type}) with {$filter_corner} Hz @ {$sampling_rate} Hz
{$coeff_data}
	`endif
endmodule

