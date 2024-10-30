//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.10.2024 14:16:08
// Copied on: 	    {$date_copy_created}
// Module Name:     FIR All-pass Filter
// Target Devices:  ASIC
//                  FPGA
// Tool Versions:   1v0
// Description:     Performing a delay filter / FIR all-pass filter on FPGA
// Processing:      Data applied on posedge clk
// Dependencies:    None
//
// State: 	        Works! (System Test done: 29.10.2024 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      BITWIDTH_DATA --> Bitwidth of input data
//                  LENGTH --> Length of used taps (= FIR filter order)
//////////////////////////////////////////////////////////////////////////////////

// Used taps order of {$filter_order} @ a sampling rate of {sampling_rate} Hz;
module FIR_DELAY_{$device_id}#(
    parameter BITWIDTH_DATA = 6'd{$bitwidth_data},
    parameter LENGTH = 10'd{$filter_order}
)(
    input wire CLK,
    input wire nRST,
    input wire EN,
    input wire START_FLAG,
    input wire [BITWIDTH_DATA-'d1:0] DATA_IN,
    output wire [BITWIDTH_DATA-'d1:0] DATA_OUT,
    output wire DATA_VALID
);
    localparam STATE_IDLE = 1'd0, STATE_PREP = 1'd1;
    
    //Control Signals
    reg state;
    reg [$clog2(LENGTH):0] cnt_adr_data;
    reg [BITWIDTH_DATA-'d1:0] taps_fir [LENGTH:0];
       
    assign DATA_VALID = (state == STATE_IDLE) && EN; 
    assign DATA_OUT = taps_fir[cnt_adr_data];
        
    //Performing FIR allpass filtering or delay line
    integer i0;
    always@(posedge CLK) begin
        if(~(nRST && EN)) begin
            state <= STATE_IDLE;
            cnt_adr_data <= 'd0;
            for(i0 = 0; i0 < LENGTH; i0 = i0 + 'd1) begin
                taps_fir[i0] = 'd0;
            end
        end else begin
            case(state)
                STATE_IDLE: begin
                    state <= (START_FLAG) ? STATE_PREP : STATE_IDLE;
                end
                STATE_PREP: begin
                    taps_fir[cnt_adr_data] <= DATA_IN;
                    state <= STATE_IDLE;
                    if(cnt_adr_data == LENGTH-'d1) begin
                        cnt_adr_data <= 'd0;
                    end else begin
                        cnt_adr_data <= cnt_adr_data + 'd1;
                    end
                end
            endcase
        end
    end
endmodule
