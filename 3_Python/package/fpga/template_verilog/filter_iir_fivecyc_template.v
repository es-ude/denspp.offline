//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     15.10.2024 13:26:51
// Copied on: 	    {$date_copy_created}
// Module Name:     IIR Filter (1st and 2nd Order / SOS-Filter, (five clock with one multiplier))
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
module Filter_IIR_FiveCyc#(
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
	`ifdef IIR_{$device_id}_USE_EXT_MULTIPLIER
        output [BITWIDTH_DATA-'d1:0] MULT_INA,
        output [BITWIDTH_WEIGHTS-'d1:0] MULT_INB,
        input [BITWIDTH_DATA+BITWIDTH_WEIGHTS-'d1:0] MULT_OUT,
    `endif
    // Data I/O
    input wire signed [BITWIDTH_DATA-'d1:0] DATA_IN,
    output wire signed [BITWIDTH_DATA-'d1:0] DATA_OUT,
    output wire DATA_VALID
);

    //################## Internal signals
    localparam UPPER_MASK = BITWIDTH_DATA + BITWIDTH_WEIGHTS - 'd3;
    localparam STATE_IDLE = 3'd6, STATE_COB0 = 3'd0, STATE_COB1 = 3'd1, STATE_COB2 = 3'd2, STATE_COA1 = 3'd3, STATE_COA2 = 3'd4, STATE_TAPS = 3'd5;

    reg [2:0] state;
    reg signed [BITWIDTH_DATA+BITWIDTH_WEIGHTS-'d1:0] pre_out;
    reg signed [BITWIDTH_DATA-'d1:0] tap_input [1:0], tap_output [1:0];

    wire signed [BITWIDTH_DATA-'d1:0] data [4:0];
    wire signed [BITWIDTH_WEIGHTS-'d1:0] coeff [4:0];

    wire signed [BITWIDTH_DATA-'d1:0] mul_ina;
    wire signed [BITWIDTH_WEIGHTS-'d1:0] mul_inb;
    wire signed [2*BITWIDTH_DATA-'d1:0] mul_out;

    //Transfer from unsigned to signed
    wire signed [BITWIDTH_DATA-'d1:0] data_in0;
    assign data_in0 = {uint_io ^ DATA_IN[BITWIDTH_DATA-'d1], DATA_IN[BITWIDTH_DATA-'d2:0]};
    assign DATA_OUT = {uint_io ^ tap_output[0][BITWIDTH_DATA-'d1], tap_output[0][BITWIDTH_DATA-'d2:0]};

    //################## Structure of Direct Form 1 Filter ##################
    assign DATA_VALID = (state == STATE_IDLE) && EN;
    assign data[0] = data_in0;
    assign data[1] = tap_input[0];
    assign data[2] = tap_input[1];
    assign data[3] = tap_output[0];
    assign data[4] = tap_output[1];

    assign mul_ina = ((state == STATE_IDLE) || (state == STATE_TAPS)) ? 'd0 : data[state];
    assign mul_inb = ((state == STATE_IDLE) || (state == STATE_TAPS)) ? 'd0 : coeff[state];

    integer i0;
    //Control-Structure
    always@(posedge CLK) begin
        if(~(nRST && EN)) begin
            state <= STATE_IDLE;
            for(i0 = 'd0; i0 < 2; i0 = i0 + 'd1) begin
                tap_input[i0] <= 'd0;
                tap_output[i0] <= 'd0;
            end
        end else begin
            case(state)
                STATE_IDLE: begin
                    state = (START_FLAG) ? STATE_COB0 : STATE_IDLE;
                end
                STATE_COB0: begin
                    state <= STATE_COB1;
                end
                STATE_COB1: begin
                    state <= STATE_COB2;
                end
                STATE_COB2: begin
                    state <= STATE_COA1;
                end
                STATE_COA1: begin
                    state <= STATE_COA2;
                end
                STATE_COA2: begin
                    state <= STATE_TAPS;
                end
                STATE_TAPS: begin
                    tap_input[0] <= data_in0;
                    tap_input[1] <= tap_input[0];
                    tap_output[0] <= pre_out[UPPER_MASK-:BITWIDTH_DATA];
                    tap_output[1] <= tap_output[0];
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
    always@(negedge CLK) begin
        if(~nRST || (state == STATE_IDLE)) begin
            pre_out <= 'd0;
        end else begin
            pre_out <= pre_out + mul_out;
        end
    end


    //################## Choicing the multiplier module ##################
    `ifdef IIR_{$device_id}_USE_LUT_MULT
        MULT_LUT_SIGNED#(BITWIDTH_DATA) MUT_B0(
            .A(mul_ina),
            .B(mul_inb),
            .Q(mul_out)
        );
    `elsif IIR_{$device_id}_USE_EXT_MULT
        assign MULT_INA = mul_ina;
        assign MULT_INB = mul_inb;
        assign mul_out = MULT_OUT;
    `else
       assign mul_out = mul_ina * mul_inb;
    `endif


    //################## Definition of the weights ##################
    // Filter coefficients input (b0, b1, b2, -a1, -a2), a0 is ignored due to 1
    `ifndef IIR_{$device_id}_USE_INT_WEIGHTS
	   genvar i0;
		for(i0 = 0; i0 < 5; i0 = i0 +1) begin
			assign coeff[i0] = FILT_WEIGHTS[i0 * BITWIDTH_DATA+:BITWIDTH_DATA];
		end
    `else
        //--- Used filter coefficients ({$filter_type}) with {$filter_corner} Hz @ {$sampling_rate} Hz
{$coeff_data}
	`endif
endmodule
