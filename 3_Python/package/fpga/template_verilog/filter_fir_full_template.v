//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     28.01.2022 14:16:08
// Copied on: 	    {$date_copy_created}
// Module Name:     FIR Filter (with One Multiplier)
// Target Devices:  ASIC (using own multiplier with //'define *_USE_OWN_MULT)
//                  FPGA (using multiplier from DSP slice)
// Tool Versions:   1v0
// Description:     Performing a FIR filtering on FPGA with custom made filter coefficients (Full implementation)
// Processing:      Data applied on posedge clk --> sampling on negedge clk
// Dependencies:    mutArrayS if using own multiplier
//
// State: 	        Works! (System Test done: 27.04.2023)
// Improvements:    Verbesserung des Timings: Aktuell wird das Timing Problem mit der zusaetzlichen Warteschleife behoben
//////////////////////////////////////////////////////////////////////////////////
{$use_int_weights}`define FIR_{$device_id}_USE_RAM_WEIGHTS
{$use_own_mult}`define FIR_{$device_id}_USE_OWN_MULT


module Filter_FIR_{$device_id}#(parameter BIT_WIDTH = 5'd{$bitwidth_data}, parameter BIT_FRAC = 5'd{$bitwidth_frac}, parameter LENGTH = 9'd{$filter_order}, parameter uint_io = 1'b{$signed_data})(
    input CLK,
    input nRST,
    input START_FLAG,
    input [BIT_WIDTH-'d1:0] DATA_IN,
    output [BIT_WIDTH-'d1:0] DATA_OUT,
	`ifndef FIR_{$device_id}_USE_RAM_WEIGHTS
		input signed wire [LENGTH * BIT_WIDTH-'d1:0] FILT_WEIGHTS;
	`endif
    output FIR_RDY
);
    localparam UPPER_MASK = BIT_WIDTH + BIT_FRAC - 5'd1;
    localparam SIZE_CNT = 8'd{$size_cnt_reg};
    localparam STATE_IDLE = 2'd0, STATE_PREP = 2'd1, STATE_FIR = 2'd2, STATE_SHIFT = 2'd3;
    
    //Control Signals
    reg cnt;
    reg [1:0] state;
    reg [SIZE_CNT-'d1:0] cnt_adr;
    reg signed [BIT_WIDTH-'d1:0] taps_fir [LENGTH-'d2:0];
    reg [2*BIT_WIDTH-'d1:0] pre_out;    
    wire [BIT_WIDTH-'d1:0] data_in0;
    wire signed [BIT_WIDTH-'d1:0] coeff_b [LENGTH-'d1:0];
    wire signed [BIT_WIDTH-'d1:0] mul_ina, mul_inb;
    
    assign FIR_RDY = (state == STATE_IDLE);    
    //Transfer from unsigned to signed
    assign data_in0 = {uint_io ^ DATA_IN[BIT_WIDTH-'d1], DATA_IN[BIT_WIDTH-'d2:0]};
    assign DATA_OUT = {uint_io ^ pre_out[UPPER_MASK], pre_out[(UPPER_MASK-'d1)-:BIT_WIDTH-'d1]};   
    
    //One Clock - Multiplier
    assign mul_ina = (state == STATE_FIR) ? ((cnt_adr == 'd0) ? data_in0 : taps_fir[cnt_adr-'d1]) : 'd0;
    assign mul_inb = (state == STATE_FIR) ? coeff_b[cnt_adr] : 'd0;
    
    //Choicing the multiplier module
    wire signed [2*BIT_WIDTH-'d1:0] mul_out;
    `ifdef FIR_{$device_id}_USE_OWN_MULT
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
    
    // Definition of the weights
	`ifndef FIR_{$device_id}_USE_RAM_WEIGHTS
		genvar i0;
		for (i0 = 'd0; i0 < LENGTH; i0 = i0 + 'd1) begin
			assign coeff_b[i0] = FILT_WEIGHTS[i0*BIT_WIDTH+:BIT_WIDTH];
		end
    `else
		//--- Used filter coefficients ({$filter_type}) with {$filter_corner} Hz @ {$sampling_rate} Hz
{$coeff_data}
	`endif
endmodule

