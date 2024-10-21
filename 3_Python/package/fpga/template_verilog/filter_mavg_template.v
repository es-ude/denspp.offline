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
// State:		    Works! (System Test done: 28.04.2023)
// Improvements:    None
//////////////////////////////////////////////////////////////////////////////////
{$use_own_multiplier}`define FIR_AVG_{$device_id}_USE_OWN_MULT


module Filter_FIR_MAVG_{$device_id}#(parameter BIT_WIDTH = 5'd{$bitwidth_data}, parameter LENGTH = 9'd{$length_mavg}, parameter uint_io = 1'b{$signed_data})(
    input CLK,
    input nRST,
    input START_FLAG,
    input [BIT_WIDTH-'d1:0] DATA_IN,
    output [BIT_WIDTH-'d1:0] DATA_OUT,
    output FIR_RDY
);
    localparam UPPER_MASK = 2 * BIT_WIDTH - 5'd1;
    localparam SIZE_CNT = 8'd{$size_cnt_reg};
    localparam STATE_IDLE = 2'd0, STATE_PREP = 2'd1, STATE_FIR = 2'd2, STATE_SHIFT = 2'd3;
    
    //Control Signals
    reg cnt;
    reg [1:0] state;
    reg [SIZE_CNT-'d1:0] cnt_adr;
    reg signed [BIT_WIDTH-'d1:0] taps_fir [LENGTH-'d2:0];
    reg [2*BIT_WIDTH-'d1:0] pre_out;    
    wire [BIT_WIDTH-'d1:0] data_in0;
    wire signed [BIT_WIDTH-'d1:0] mul_ina, mul_inb;

    wire [BIT_WIDTH-'d1:0] coeff_avg;
    assign coeff_avg = {$bitwidth_data}'d{$coeff_mavg_dec}; //coeff_avg = {$coeff_mavg_float} = {$coeff_mavg_hex}
    
    assign FIR_RDY = (state == STATE_IDLE);    
    //Transfer from unsigned to signed
    assign data_in0 = {uint_io ^ DATA_IN[BIT_WIDTH-'d1], DATA_IN[BIT_WIDTH-'d2:0]};
    assign DATA_OUT = {uint_io ^ pre_out[UPPER_MASK], pre_out[(UPPER_MASK-'d1)-:BIT_WIDTH-'d1]};   
    
    //One Clock - Multiplier
    assign mul_ina = (state == STATE_FIR) ? ((cnt_adr == 'd0) ? data_in0 : taps_fir[cnt_adr-'d1]) : 'd0;
    assign mul_inb = (state == STATE_FIR) ? coeff_b : 'd0;
    
    //Choicing the multiplier module
    wire signed [2*BIT_WIDTH-'d1:0] mul_out;
    `ifdef FIR_AVG_{$device_id}_USE_OWN_MULT
        mutArrayS#(BIT_WIDTH) MUT(
            .A(mul_ina),
            .B(mul_inb),
            .Q(mul_out)
        );
    `else
       assign mul_out = mul_ina * mul_inb;
    `endif
        
    //Performing FIR
    integer i0;
    always@(posedge CLK) begin
        if(~nRST) begin
            // --- Init/RST Phase
            state <= STATE_IDLE;
            cnt_adr <= 'd0;
            for(i0 = 0; i0 < LENGTH-'d1; i0 = i0 + 'd1) begin
                taps_fir[i0] = 'd0;
            end
            pre_out <= 'd0;
            cnt <= 1'd0;
        end else begin
            case(state)
                STATE_IDLE: begin
                    state <= (START_FLAG) ? STATE_PREP : STATE_IDLE;
                end
                STATE_PREP: begin
                    state <= STATE_FIR;
                    pre_out <= 'd0;
                end
                STATE_FIR: begin
                    // --- Calculation Phase
                    if(cnt_adr == LENGTH-'d1) begin
                        cnt_adr <= 'd0;
                        state <= STATE_SHIFT;
                        cnt <= 1'd0;
                    end else begin 
                        cnt_adr <= (cnt == 1'd1) ? cnt_adr + 'd1 : cnt_adr;
                        state <= (cnt == 1'd1) ? STATE_FIR : state;
                    end
                    pre_out <= pre_out + ((cnt == 1'd1) ? mul_out : 'd0);
                    cnt <= ~cnt;
                end
                STATE_SHIFT: begin
                    cnt <= 1'd0;
                    for(i0 = 'd0; i0 < LENGTH-'d1; i0 = i0 + 'd1) begin
                        taps_fir[i0] <= (i0 == 'd0) ? data_in0 : taps_fir[i0-'d1];
                    end
                    state <= STATE_IDLE;
                end
            endcase
        end
    end
endmodule

