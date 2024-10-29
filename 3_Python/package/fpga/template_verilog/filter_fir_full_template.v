//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     15.10.2024 14:16:08
// Copied on: 	    {$date_copy_created}
// Module Name:     FIR Filter (with One Multiplier)
// Target Devices:  ASIC (using own multiplier with //'define *_USE_OWN_MULT)
//                  FPGA (using multiplier from DSP slice)
// Tool Versions:   1v0
// Description:     Performing a FIR filtering on FPGA with custom made filter coefficients (Full implementation)
// Processing:      Data applied on posedge clk --> sampling on negedge clk
// Dependencies:    mutArrayS if using own multiplier
//
// State: 	        Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH_DATA --> Bitwidth of input data
//                  LENGTH --> Length of used taps (=FIR filter order)
//                  uint_io --> Do type conversion (1'b1) from unsigned to signed to unsigned else don't do
//                  BITWIDTH_WEIGHTS --> Bitwidth of all weights (fraction is bitwidth)
//////////////////////////////////////////////////////////////////////////////////
{$use_int_weights}`define FIR_{$device_id}_USE_RAM_WEIGHTS
{$use_lut_mult}`define FIR_{$device_id}_USE_LUT_MULT
{$use_ext_mult}`define FIR_{$device_id}_USE_EXT_MULT


// Input values are integer or unsigned with size of BITWIDTH_DATA (no fixed point)
// Internal operation with signed values and all weights have fraction width of BITWIDTH_DATA;
module Filter_FIR_{$device_id}#(
	parameter BITWIDTH_DATA = 6'd{$bitwidth_data},
	parameter LENGTH = 10'd{$filter_order}, 
	parameter uint_io = 1'b{$signed_data},
	parameter BITWIDTH_WEIGHTS = 6'd{$bitwidth_weights}
)(
    input wire CLK,
    input wire nRST,
    input wire EN,
    input wire START_FLAG,
    input wire [BITWIDTH_DATA-'d1:0] DATA_IN,
    output wire [BITWIDTH_DATA-'d1:0] DATA_OUT,
	// Filter coefficients input (b0, b1, b2, ..., bN)
	`ifndef FIR_{$device_id}_USE_RAM_WEIGHTS
		input wire signed [LENGTH * BITWIDTH_WEIGHTS-'d1:0] FILT_WEIGHTS,
	`endif
    `ifdef FIR_{$device_id}_USE_EXT_MULT
        output wire [BITWIDTH_DATA-'d1:0] MULT_INA,
        output wire [BITWIDTH_WEIGHTS-'d1:0] MULT_INB,
        input wire [BITWIDTH_DATA+BITWIDTH_WEIGHTS-'d1:0] MULT_OUT,
    `endif
    output wire DATA_VALID
);
    localparam UPPER_MASK = BITWIDTH_DATA + BITWIDTH_WEIGHTS;
    localparam STATE_IDLE = 2'd0, STATE_PREP = 2'd1, STATE_CALC = 2'd2;
    
    //Control Signals
    reg [1:0] state;
    reg [$clog2(LENGTH)-'d1:0] cnt_adr_wght;
    reg [$clog2(LENGTH)-'d1:0] cnt_adr_data;
    reg signed [BITWIDTH_DATA-'d1:0] taps_fir [LENGTH-'d1:0];
    reg [BITWIDTH_DATA+BITWIDTH_WEIGHTS-'d1:0] mac_out;
    
    wire [BITWIDTH_DATA-'d1:0] data_in0;
    wire signed [BITWIDTH_WEIGHTS-'d1:0] coeff_b [LENGTH-'d1:0];
    wire signed [BITWIDTH_DATA-'d1:0] mul_ina;
    wire signed [BITWIDTH_WEIGHTS-'d1:0] mul_inb;
    
    assign DATA_VALID = (state == STATE_IDLE) && EN;    
    //Transfer from unsigned to signed
    assign data_in0 = {uint_io ^ DATA_IN[BITWIDTH_DATA-'d1], DATA_IN[BITWIDTH_DATA-'d2:0]};
    assign DATA_OUT = {uint_io ^ mac_out[UPPER_MASK-'d1], mac_out[(UPPER_MASK-'d2)-:BITWIDTH_DATA-'d1]};   
    //One Clock - Multiplier
    assign mul_ina = (state == STATE_CALC) ? taps_fir[cnt_adr_data] : 'd0;
    assign mul_inb = (state == STATE_CALC) ? coeff_b[cnt_adr_wght] : 'd0;
    
    //Choicing the multiplier module
    wire signed [BITWIDTH_DATA+BITWIDTH_WEIGHTS-'d1:0] mul_out;
    `ifdef FIR_{$device_id}_USE_LUT_MULT   
        mutArrayS#(BITWIDTH_DATA) MUT(
            .A(mul_ina),
            .B(mul_inb),
            .Q(mul_out)
        );
    `elsif FIR_{$device_id}_USE_EXT_MULT
        assign MULT_INA = mul_ina;
        assign MULT_INB = mul_inb;
        assign mul_out = MULT_OUT;
    `else
       assign mul_out = mul_ina * mul_inb;
    `endif
        
     //Performing FIR
    integer i0;
    always@(posedge CLK) begin
        if(~(nRST && EN)) begin
            state <= STATE_IDLE;
            cnt_adr_data <= 'd0;
            cnt_adr_wght <= 'd0;
            for(i0 = 0; i0 < LENGTH; i0 = i0 + 'd1) begin
                taps_fir[i0] = 'd0;
            end
        end else begin
            case(state)
                STATE_IDLE: begin
                    state <= (START_FLAG) ? STATE_PREP : STATE_IDLE;
                end
                STATE_PREP: begin
                    taps_fir[cnt_adr_data] <= data_in0;
                    state <= STATE_CALC;
                end
                STATE_CALC: begin
                    //Cnt structure for weight counter
                    if(cnt_adr_wght == LENGTH-'d1) begin
                        cnt_adr_wght <= 'd0;
                        state <= STATE_IDLE;
                        cnt_adr_data <= cnt_adr_data;
                    end else begin 
                        cnt_adr_wght <= cnt_adr_wght + 'd1;
                        state <= STATE_CALC;
                        //Cnt structure for data counter
                        if(cnt_adr_data == 'd0) begin
                            cnt_adr_data <= LENGTH-'d1;
                        end else begin 
                            cnt_adr_data <= cnt_adr_data - 'd1;
                        end 
                    end
                end
            endcase
        end
    end      
    always@(negedge CLK) begin
        if(~nRST || (state == STATE_PREP)) begin
            mac_out <= 'd0;
        end else begin
            mac_out <= mac_out + ((state == STATE_CALC) ? mul_out : 'd0);
        end
    end
    
    // Definition of the weights
	`ifndef FIR_{$device_id}_USE_RAM_WEIGHTS
		genvar i1;
	    for (i1 = 'd0; i1 < LENGTH; i1 = i1 + 'd1) begin
            assign coeff_b[i1] = FILT_WEIGHTS[i1*BITWIDTH_WEIGHTS+:BITWIDTH_WEIGHTS];
        end
    `else
		//--- Used filter coefficients ({$filter_type}) with {$filter_corner} Hz @ {$sampling_rate} Hz
{$coeff_data}
	`endif
endmodule

