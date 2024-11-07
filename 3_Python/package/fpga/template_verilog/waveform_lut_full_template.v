// ---------------------------------------------------
// Company: UDE-ES
// Design Name: SineLUT Generator
// Generate file on: 10/30/2024, 12:55:55
// Original sinusoidal frequency = {f_sine / 1e3: .1f} kHz
// Size of SineLUT array: {$num_sinelut} ({$bitsize_lut} bit)
// ---------------------------------------------------
{$do_read_lut_external}`define LUT{$device_id}_ACCESS_EXTERNAL
{$do_cnt_external}`define LUT{$device_id}_COUNT_EXTERNAL

// --- CODE FOR READING DATA FROM EXTERNAL
// wire {$signed_type} [{$num_sinelut} * {$bitsize_lut} - 'd1:0] LUT_ROM = {{$lut_data_stream}};


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
	`ifdef LUT{$device_id}_COUNT_EXTERNAL
	    input wire [$clog2(LUT_WIDTH)-'d1:0] WAIT_CYC,
	`endif
	`ifdef LUT{$device_id}_ACCESS_INTERNAL
	    input wire {$signed_type} [BIT_WIDTH * LUT_WIDTH - 'd1:0] LUT_ROM,
	`endif
	output wire {$signed_type} [BIT_WIDTH-'d1:0] LUT_VALUE,
	output wire LUT_END
);

    reg [$clog2(LUT_WIDTH)-'d1:0] cnt_sine;
    `ifdef LUT{$device_id}_COUNT_EXTERNAL
        reg [WAIT_WIDTH-'d1:0] cnt_wait;
    `else
        reg [$clog2(WAIT_CYC)-'d1:0] cnt_wait;
    `endif

    // --- Processing LUT data
    assign LUT_END = (cnt_sine == (LUT_WIDTH-'d1));
    wire {$signed_type} [BIT_WIDTH-'d1:0] lut_ram [LUT_WIDTH-'d1:0];
    assign LUT_VALUE = lut_ram[cnt_sine];
    `ifdef LUT{$device_id}_ACCESS_EXTERNAL
        integer i0;
        for(i0 = 'd0; i0 < LUT_WIDTH; i0 = i0 + 'd1) begin
            lut_ram[i0] = LUT_ROM[i0+:BIT_WIDTH];
        end
    `else
        // --- Data save in BRAM
{$lut_data_array}
    `endif

    // --- Counter for Full LUT Reading
    always@(posedge CLK_SYS) begin
        if(~(nRST && EN)) begin
            cnt_sine <= 'd0;
            cnt_wait <= 'd0;
        end else begin
            if(cnt_wait == WAIT_CYC-'d1) begin
                cnt_sine <= (cnt_sine == LUT_WIDTH-'d1) ? 'd1 : cnt_sine + 'd1;
                cnt_wait <= 'd0;
            end else begin
                cnt_sine <= cnt_sine;
                cnt_wait <= cnt_wait + 'd1;
            end
        end
    end

endmodule
