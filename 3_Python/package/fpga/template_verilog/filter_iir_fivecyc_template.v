//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     31.01.2022 13:26:44
// Copied on: 	    {$date_copy_created}
// Module Name:     IIR Filter (1st and 2nd Order / SOS-Filter, (five clock with one multiplier))
// Target Devices:  ASIC (Implementing and using the mutArrayS module)
//                  FPGA (Using DSP block for multiplication)
// Tool Versions:   1v0
// Description:     Structure: Direct Form 2, signed integer operation
// Processing:      Data applied on posedge clk --> sampling on negedge clk
// Dependencies:    mutArrayS with custom-made multiplier
// 
// State:		    Works! (System Test done: 28.04.2023)
// Improvements:    Verbesserung des Timings: Aktuell wird das Timing Problem mit der zusaetzlichem Takt behoben
//////////////////////////////////////////////////////////////////////////////////
{$use_ram_coeff}`define IIR_{$device_id}_USE_RAM_COEFF
{$use_own_mult}`define IIR_{$device_id}_USE_OWN_MULT


module Filter_IIR_{$device_id}_FiveCyc#(parameter BIT_WIDTH = 4'd{$bitwidth_data}, parameter BIT_FRAC = 4'd{$bitwidth_frac}, parameter uint_io = 1'b{$signed_data})(
    input wire CLK,
    input wire nRST,
    input wire START_FLAG,
    //--- Filter coefficients input
	`ifndef IIR_{$device_id}_USE_RAM_COEFF
		input wire signed [3* BIT_WIDTH-'d1:0] FILT_WEIGHTSA,
		input wire signed [3* BIT_WIDTH-'d1:0] FILT_WEIGHTSB,
	`endif
    //--- Data I/O
    input wire signed [BIT_WIDTH-'d1:0] DATA_IN,
    output wire signed [BIT_WIDTH-'d1:0] DATA_OUT,
    output wire IIR_RDY
);

    localparam UPPER_MASK = BIT_WIDTH + BIT_FRAC - 'd1;
    localparam STATE_IDLE = 3'd0, STATE_COA1 = 3'd1, STATE_COA2 = 3'd2, STATE_COB0 = 3'd3, STATE_COB1 = 3'd4, STATE_COB2 = 3'd5, STATE_TAPS = 3'd6;
    
    reg [2:0] state;
    assign IIR_RDY = (state == STATE_IDLE);    
    
    //Multiplier unit
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
    reg clk_iir_int;
    reg signed [2*BIT_WIDTH-'d1:0] pre_out;
    reg signed [BIT_WIDTH-'d1:0] tap_input [1:0], tap_output [1:0];
    
    reg signed [BIT_WIDTH-'d1:0] mul_ina, mul_inb;
    wire signed [BIT_WIDTH-'d1:0] data_in0;
    
    //Transfer from unsigned to signed
    assign data_in0 = {uint_io ^ DATA_IN[BIT_WIDTH-'d1], DATA_IN[BIT_WIDTH-'d2:0]};
    assign DATA_OUT = {uint_io ^ tap_output[0][BIT_WIDTH-'d1], tap_output[0][BIT_WIDTH-'d2:0]};   
    
    //Choicing the multiplier module
    wire signed [2*BIT_WIDTH-'d1:0] mul_out;
    `ifdef IIR_{$device_id}_USE_OWN_MULT
        mutArrayS#(BIT_WIDTH) MUT_B0(
            .A(mul_ina),
            .B(mul_inb),
            .Q(mul_out)
        );
    `else
       assign mul_out = mul_ina * mul_inb;
    `endif
    
    //Control-Structure
    always@(posedge CLK) begin
        if(~nRST) begin
            state <= STATE_IDLE;
            tap_input[0] <= 'd0;
            tap_input[1] <= 'd0;
            tap_output[0] <= 'd0;
            tap_output[1] <= 'd0;
            pre_out <= 'd0;
            clk_iir_int <= 'd0;
            mul_ina <= 'd0;
            mul_inb <= coeff_a[0];
        end else begin 
            clk_iir_int <= (state == STATE_IDLE && state == STATE_TAPS) ? 1'b0 : ~clk_iir_int;
            case(state)
                STATE_IDLE: begin
                    state = (START_FLAG) ? STATE_COA1 : STATE_IDLE;
                end
                STATE_COA1: begin
                    state <= (clk_iir_int) ? STATE_COA2 : state;
                    pre_out <= (clk_iir_int) ? mul_out : pre_out;
                    mul_ina <= (~clk_iir_int) ? tap_output[0] : mul_ina;
                    mul_inb <= (~clk_iir_int) ? coeff_a[1] : mul_inb;
                end
                STATE_COA2: begin
                    state <= (clk_iir_int) ? STATE_COB0 : state;
                    pre_out <= (clk_iir_int) ?  mul_out : pre_out;
                    mul_ina <= (~clk_iir_int) ? tap_output[1] : mul_ina;
                    mul_inb <= (~clk_iir_int) ? coeff_a[2] : mul_inb;
                end
                STATE_COB0: begin
                    state <= (clk_iir_int) ? STATE_COB1 : state;
                    pre_out <= (clk_iir_int) ? mul_out : pre_out;
                    mul_ina <= (~clk_iir_int) ? data_in0 : mul_ina;
                    mul_inb <= (~clk_iir_int) ? coeff_b[0] : mul_inb;
                end
                STATE_COB1: begin
                    state <= (clk_iir_int) ? STATE_COB2 : state;
                    pre_out <= (clk_iir_int) ? mul_out : pre_out;
                    mul_ina <= (~clk_iir_int) ? tap_input[0] : mul_ina;
                    mul_inb <= (~clk_iir_int) ? coeff_b[1] : mul_inb;
                end
                STATE_COB2: begin
                    state <= (clk_iir_int) ? STATE_TAPS : state;
                    pre_out <= (clk_iir_int) ? mul_out : pre_out;
                    mul_ina <= (~clk_iir_int) ? tap_input[1] : mul_ina;
                    mul_inb <= (~clk_iir_int) ? coeff_b[2] : mul_inb;
                end
                STATE_TAPS: begin
                    tap_input[0] <= data_in0;
                    tap_input[1] <= tap_input[0];
                    tap_output[0] <= pre_out[UPPER_MASK-:BIT_WIDTH];
                    tap_output[1] <= tap_output[0];
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
endmodule
