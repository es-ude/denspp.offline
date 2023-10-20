// ---------------------------------------------------
// Raw data converting from DeNSSP to Verilog (v1.0)
// Testbench file for playing data sets in simulations
// ---------------------------------------------------
// Generate file on: 09/15/2023, 09:27:00
// Original sampling rate fs =  20.00 kHz
// Size of input array: 5000
// ---------------------------------------------------
// Code Example for Implementation in Testbench:
//
//	NeuralSim_Mem DataSim(
//		.CLK_ADC(clk0),
//		.nRST(reset_n),
//		.EN(enable_sim)
//		.DATA_OUT(data_sim),
//		.TRGG_OUT(trgg_sim),
//		.DATA_END(sim_end)
//	);
// ---------------------------------------------------

module NeuralSim_Mem(
	input wire CLK_ADC,
	input wire nRST,
	input wire EN,
	output wire signed [11:0] DATA_OUT,
	output wire TRGG_OUT,
	output wire DATA_END
);

reg [12:0] cnt_pos;

reg bram_trgg [0:4999];
assign TRGG_OUT = (EN && !DATA_END) ? bram_trgg[cnt_pos] : 1'd0;
reg signed [11:0] bram_data [0:4999];
assign DATA_OUT = (EN && !DATA_END) ? bram_data[cnt_pos] : 12'd0;

assign DATA_END = (cnt_pos == 'd4999);

initial begin
	$readmemh("data_raw.mem", bram_data);
	$readmemb("data_trg.mem", bram_trgg);
end

always@(posedge CLK_ADC or negedge nRST) begin
	if(!nRST) begin
		cnt_pos = 13'd0;
	end else begin
		cnt_pos = cnt_pos + ((EN && !DATA_END) ? 13'd1 : 13'd0);
	end
end

endmodule
