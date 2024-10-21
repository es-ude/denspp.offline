//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     31.01.2022 12:26:47
// Copied on: 	    {$date_copy_created}
// Module Name:     IIR Filter (1st and 2nd Order / SOS-Filter, One Cycle with five MULT)
// Target Devices:  ASIC (Implementing and using the mutArrayS module)
//                  FPGA (Using DSP block for multiplication)
// Tool Versions:   1v0
// Description:     Structure: Direct Form 2, signed integer operation
// Processing:      Data applied on posedge clk --> sampling on negedge clk
// Dependencies:    mutArrayS with custom-made multiplier
// 
// State: 	        Works! (System Test done: 27.04.2023)
// Improvements:    None
//////////////////////////////////////////////////////////////////////////////////
{$use_ram_coeff}`define IIR_{$device_id}_USE_RAM_COEFF
{$use_own_mult}`define IIR_{$device_id}_USE_OWN_MULT

module Filter_IIR_{$device_id}_OneCyc#(parameter BIT_WIDTH = 5'd{$bitwidth_data}, parameter BIT_FRAC = 5'd{$bitwidth_frac}, parameter uint_io = 1'b{$signed_data})(
    input wire CLK,
    input wire nRST,
    input wire START_FLAG,
    //--- Filter coefficients input
	`ifndef IIR_{$device_id}_USE_RAM_COEFF
		input wire signed [3 * BIT_WIDTH-'d1:0] FILT_WEIGHTSA,
		input wire signed [3 * BIT_WIDTH-'d1:0] FILT_WEIGHTSB,
	`endif
    //--- Data I/O
    input wire signed [BIT_WIDTH-'d1:0] DATA_IN,
    output wire signed [BIT_WIDTH-'d1:0] DATA_OUT,
    output wire IIR_RDY
);
    localparam UPPER_MASK = BIT_WIDTH + BIT_FRAC - 'd1;

	wire signed [BIT_WIDTH-'d1:0] coeff_a [2:0];
	wire signed [BIT_WIDTH-'d1:0] coeff_b [2:0];
	`ifndef IIR_{$device_id}_USE_RAM_COEFF
		genvar i0;
		for(i0 = 0; i0 < 3; i0 = i0 +1) begin
			assign coeff_a[i0] = FILT_WEIGHTSA[i0 * BIT_WIDTH+:BIT_WIDTH];
			assign coeff_b[i0] = FILT_WEIGHTSB[i0 * BIT_WIDTH+:BIT_WIDTH];
		end
	`else
		//--- Used filter coefficients for ({$filter_type}) with [{$filter_corner}] Hz @ {$sampling_rate} Hz
{$coeffa_data}
{$coeffb_data}
	`endif
    
    //################## Structure of DF1 Filter ##################
    //Internal signals
    reg signed [BIT_WIDTH-'d1:0] tap_input [1:0], tap_output [1:0];
    wire signed [2*BIT_WIDTH-'d1:0] step_input [2:0], step_output [1:0];
    wire signed [2*BIT_WIDTH-'d1:0] dout [1:0], out;
    wire signed [BIT_WIDTH-'d1:0] data_in0;
    
    assign dout[0] = step_input[0] + step_input[1] + step_input[2];
    assign dout[1] = step_output[0] + step_output[1];
    assign out = dout[0] + dout[1];
    assign IIR_RDY = !START_FLAG;
    
    assign data_in0 = {uint_io ^ DATA_IN[BIT_WIDTH-'d1], DATA_IN[BIT_WIDTH-'d2:0]};
    assign DATA_OUT = {uint_io ^ tap_output[0][BIT_WIDTH-'d1], tap_output[0][BIT_WIDTH-'d2:0]};
    
    //Control-Structure
    always@(posedge CLK) begin
        if(!nRST) begin
            tap_input[0] <= 'd0;
            tap_input[1] <= 'd0;
            tap_output[0] <= 'd0;
            tap_output[1] <= 'd0;
        end else begin 
            tap_input[0] <= (START_FLAG) ? data_in0 : tap_input[0];
            tap_input[1] <= (START_FLAG) ? tap_input[0] : tap_input[1];
            tap_output[0] <= (START_FLAG) ? out[UPPER_MASK-:BIT_WIDTH] : tap_output[0];
            tap_output[1] <= (START_FLAG) ? tap_output[0] : tap_output[1];
        end
    end
    
    //Using the selected multiplier module
    `ifdef IIR_{$device_id}_USE_OWN_MULT
        mutArrayS#(BIT_WIDTH) MUT_B0(.A(data_in0),      .B(coeff_b[0]), .Q(step_input[2]));
        mutArrayS#(BIT_WIDTH) MUT_B1(.A(tap_input[0]),  .B(coeff_b[1]), .Q(step_input[0]));
        mutArrayS#(BIT_WIDTH) MUT_B2(.A(tap_input[1]),  .B(coeff_b[2]), .Q(step_input[1]));
        mutArrayS#(BIT_WIDTH) MUT_A1(.A(tap_output[0]), .B(coeff_a[1]), .Q(step_output[0]));
        mutArrayS#(BIT_WIDTH) MUT_A2(.A(tap_output[1]), .B(coeff_a[2]), .Q(step_output[1]));
    `else
        assign step_input[2] = data_in0 * coeff_b[0];
        assign step_input[0] = tap_input[0] * coeff_b[1];
        assign step_input[1] = tap_input[1] * coeff_b[2];
        assign step_output[0] = tap_output[0] * coeff_a[1];
        assign step_output[1] = tap_output[1] * coeff_a[2];
    `endif
endmodule

