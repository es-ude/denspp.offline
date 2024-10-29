//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date: 	21.10.2024 12:38:44
// Copied on: 	    {$date_copy_created}
// Module Name:     FIR (OneMultiplier) for Moving Average
// Target Devices:  ASIC (using own multiplier with //'define *_USE_OWN_MULT)
//                  FPGA (using multiplier from DSP slice)
// Tool Versions:   1v0
// Description:     Moving Average with N = {$length_mavg} @ fs = {$sampling_rate} Hz
// Processing:      Data applied on posedge clk --> sampling on negedge clk
// Dependencies:    mutArrayS if using own multiplier
//
// State:		    Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH_DATA --> Bitwidth of input data
//                  LENGTH --> Length of used taps (=FIR filter order)
//                  uint_io --> Do type conversion (1'b1) from unsigned to signed to unsigned else don't do
//                  BITWIDTH_WEIGHTS --> Bitwidth of all weights (fraction is bitwidth)
//////////////////////////////////////////////////////////////////////////////////
{$use_own_mult}`define MAVG_{$device_id}_USE_OWN_MULT
{$use_ext_mult}`define MAVG_{$device_id}_USE_EXT_MULT


// Internal operation with signed values and scaling weight has fraction width of bitwidth (BITFRAC_WEIGHTS = BITWIDTH_WEIGHT);
module Filter_MAVG_{$device_id}#(
    parameter BITWIDTH_DATA = 6'd{$bitwidth_data},
    parameter LENGTH = 9'd{$length_mavg},
    parameter uint_io = 1'b{$signed_data},
    parameter BITWIDTH_WEIGHT = 6'd{$bitwidth_data}
)(
    input wire CLK,
    input wire nRST,
    input wire EN,
    input wire START_FLAG,
    input wire [BITWIDTH_DATA-'d1:0] DATA_IN,
    output wire [BITWIDTH_DATA-'d1:0] DATA_OUT,
    `ifdef MAVG_{$device_id}_USE_EXT_MULTIPLIER
        output wire [BITWIDTH_DATA-'d1:0] MULT_INA,
        output wire [BITWIDTH_WEIGHT-'d1:0] MULT_INB,
        input wire [BITWIDTH_DATA+BITWIDTH_WEIGHT-'d1:0] MULT_OUT,
    `endif
    output wire DATA_VALID
);
    localparam UPPER_MASK = BITWIDTH_DATA + BITWIDTH_WEIGHT;
    localparam SCALE_VAL = (2**BITWIDTH_WEIGHT) / LENGTH;
    localparam STATE_IDLE = 2'd0, STATE_PREP = 2'd1, STATE_CALC = 2'd2;

    //Control Signals
    reg [1:0] state;
    reg [$clog2(LENGTH)-'d1:0] cnt_adr_data;
    reg [$clog2(LENGTH)-'d1:0] cnt_ite;
    reg signed [BITWIDTH_DATA-'d1:0] taps_fir [LENGTH-'d1:0];
    reg [(BITWIDTH_DATA+BITWIDTH_WEIGHT)-'d1:0] pre_out;

    wire [BITWIDTH_DATA-'d1:0] data_in0;
    wire signed [BITWIDTH_DATA-'d1:0] mul_ina;
    wire signed [BITWIDTH_WEIGHT-'d1:0] coeff_b, mul_inb;
	assign coeff_b = SCALE_VAL;

    assign DATA_VALID = (state == STATE_IDLE) && EN;
    //Transfer from unsigned to signed
    assign data_in0 = {uint_io ^ DATA_IN[BITWIDTH_DATA-'d1], DATA_IN[BITWIDTH_DATA-'d2:0]};
    assign DATA_OUT = {uint_io ^ pre_out[UPPER_MASK-'d1], pre_out[(UPPER_MASK-'d2)-:BITWIDTH_DATA-'d1] + pre_out[BITWIDTH_WEIGHT]};

    //One Clock - Multiplier
    assign mul_ina = (state == STATE_CALC) ? taps_fir[cnt_adr_data] : 'd0;
    assign mul_inb = (state == STATE_CALC) ? coeff_b : 'd0;

    //Choicing the multiplier mode
    wire signed [(BITWIDTH_DATA+BITWIDTH_WEIGHT)-'d1:0] mul_out;
    `ifdef MAVG_0_USE_LUT_MULTIPLIER
        mutArrayS#(BITWIDTH_DATA) MUT(
            .A(mul_ina),
            .B(mul_inb),
            .Q(mul_out)
        );
    `elsif MAVG_{$device_id}_USE_EXT_MULTIPLIER
        assign MULT_INA = mul_ina;
        assign MULT_INB = mul_inb;
        assign mul_out = MULT_OUT;
    `else
       assign mul_out = mul_ina * mul_inb;
    `endif

    //Performing computation
    integer i0;
    always@(posedge CLK) begin
        if(~(nRST && EN)) begin
            state <= STATE_IDLE;
            cnt_adr_data <= 'd0;
            cnt_ite <= 'd0;
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
                    // Iteration counter
                    if(cnt_ite == LENGTH -'d1) begin
                        cnt_ite <= 'd0;
                        state <= STATE_IDLE;
                        cnt_adr_data <= cnt_adr_data;
                    end else begin
                        cnt_ite <= cnt_ite + 'd1;
                        state <= STATE_CALC;
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
            pre_out <= 'd0;
        end else begin
            pre_out <= pre_out + ((state == STATE_CALC) ? mul_out : 'd0);
        end
    end
endmodule
