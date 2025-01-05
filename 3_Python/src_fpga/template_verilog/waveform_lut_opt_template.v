//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	21.10.2024 12:38:44
// Copied on: 	    {$date_copy_created}
// Module Name:     LUT Generator for Storing/Calling Optimized (Quarter) Waveforms
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v1
// Description:     Digital Direct Syntheziser with Analog Signal Waveforms ({$num_sinelut} x {$bitsize_lut} bit)
// Dependencies:    None
//
// State:		    Works! (System Test done: 07.11.2024 on Arty A7-35T with 22% usage)
// Improvements:    None
// Parameters:      BIT_WIDTH   --> Bitwidth of the output value
//                  LUT_WIDTH   --> Length of LUT for saving waveform
//                  WAIT_CYC    --> Number of clock cycles to wait before update output
//                  WAIT_WIDTH  --> Bitwidth for defining WAIT_CYC with external middleware (R/nW)
//////////////////////////////////////////////////////////////////////////////////
{$do_read_lut_external}`define LUT{$device_id}_ACCESS_EXTERNAL
{$do_cnt_external}`define LUT{$device_id}_COUNT_EXTERNAL
{$do_trgg_external}`define LUT{$device_id}_TRGG_EXTERNAL

// --- CODE FOR READING DATA FROM EXTERNAL
// wire {$signed_type} [{$num_sinelut} * ({$bitsize_lut}-'d1) - 'd1:0] LUT_ROM;
// assign LUT_ROM = {{$lut_data_stream}};


module LUT_WVF_GEN{$device_id}#(
	parameter LUT_WIDTH = 10'd{$num_sinelut},
	parameter BIT_WIDTH = 6'd{$bitsize_lut},
	`ifndef LUT{$device_id}_COUNT_EXTERNAL
        parameter WAIT_CYC = 12'd{$wait_cycles}
    `else
        parameter WAIT_WIDTH = 10'd{$wait_cnt_width}
	`endif
)(
	input wire CLK_SYS,
	input wire nRST,
	input wire EN,
	`ifdef LUT{$device_id}_TRGG_EXTERNAL
	    input wire TRGG_CNT_FLAG,
    `elsif LUT{$device_id}_COUNT_EXTERNAL
        input wire [$clog2(LUT_WIDTH)-'d1:0] WAIT_CYC,
    `endif
    `ifdef LUT{$device_id}_ACCESS_EXTERNAL
        input wire {$signed_type} [(BIT_WIDTH-'d1) * LUT_WIDTH - 'd1:0] LUT_ROM,
    `endif
	output wire {$signed_type} [BIT_WIDTH-'d1:0] LUT_VALUE,
	output wire LUT_END
);

    // --- Registers for counting and controlling
    wire increase_cnt_sine;
    reg [$clog2(LUT_WIDTH)-'d1:0] cnt_sine;
    reg [1:0] cnt_phase;
    `ifndef LUT{$device_id}_TRGG_EXTERNAL
        `ifdef LUT{$device_id}_COUNT_EXTERNAL
            reg [WAIT_WIDTH-'d1:0] cnt_wait;
        `else
            reg [$clog2(WAIT_CYC)-'d1:0] cnt_wait;
        `endif
        // --- Counter for Downsampling System Clock
        always@(posedge CLK_SYS) begin
            if(~(nRST && EN)) begin
                cnt_wait <= 'd0;
            end else begin
                if(cnt_wait == WAIT_CYC-'d1) begin
                    cnt_wait <= 'd0;
                end else begin
                    cnt_wait <= cnt_wait + 'd1;
                end
            end
        end
        assign increase_cnt_sine = (cnt_wait == WAIT_CYC-'d1);
    `else
        assign increase_cnt_sine = TRGG_CNT_FLAG;
    `endif

    // --- Processing LUT data
    assign LUT_END = (cnt_sine == (LUT_WIDTH-'d1)) && (cnt_phase == 2'd3);
    wire {$signed_type} [BIT_WIDTH-'d2:0] lut_rom_int [LUT_WIDTH-'d1:0];
    //Unsigned Processing
    {$do_unsigned_call}assign LUT_VALUE = (cnt_phase == 2'd0) ? {1'd1, lut_rom_int[cnt_sine]} : ((cnt_phase == 2'd1) ? {1'd1, lut_rom_int[LUT_WIDTH-cnt_sine-'d1]} : ((cnt_phase == 2'd2) ? {1'd0, {(BIT_WIDTH-'d2){1'd1}}} - lut_rom_int[cnt_sine] : ({1'd0, {(BIT_WIDTH-'d2){1'd1}}} - lut_rom_int[LUT_WIDTH-cnt_sine-'d1])));
    //Signed Processing
    {$do_signed_call}assign LUT_VALUE = (cnt_phase == 2'd0) ? {1'd0, lut_rom_int[cnt_sine]} : ((cnt_phase == 2'd1) ? {1'd0, lut_rom_int[LUT_WIDTH-cnt_sine-'d1]} : ((cnt_phase == 2'd2) ? {1'd1, -lut_rom_int[cnt_sine]-'d1} : ({1'd1, -lut_rom_int[LUT_WIDTH-cnt_sine-'d1]-'d1})));
    `ifdef LUT{$device_id}_ACCESS_EXTERNAL
        genvar i0;
        for(i0 = 'd0; i0 < LUT_WIDTH; i0 = i0 + 'd1) begin
            assign lut_rom_int[i0] = LUT_ROM[i0*(BIT_WIDTH-'d1)+:BIT_WIDTH-'d1];
        end
    `else
        // --- Data save in BRAM
{$lut_data_array}
    `endif

    //--- Counter for Quarter Wave Reading (Symmetric)
    always@(posedge CLK_SYS or negedge nRST) begin
        if(~(nRST && EN)) begin
            cnt_phase <= 2'd0;
            cnt_sine <= 'd0;
        end else begin
            if(increase_cnt_sine) begin
                cnt_phase <= cnt_phase + ((cnt_sine == LUT_WIDTH-'d1) ? 2'd1 : 2'd0);
                cnt_sine <= (cnt_sine == (LUT_WIDTH-'d1)) ? 'd1 : cnt_sine + 'd1;
            end else begin
                cnt_phase <= cnt_phase;
                cnt_sine <= cnt_sine;
            end
        end
    end
endmodule
