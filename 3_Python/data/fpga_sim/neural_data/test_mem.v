// ---------------------------------------------------
// Raw data converting from DeNSSP to Verilog (v1.0)
// Testbench file for playing data sets in simulations
// ---------------------------------------------------
// Generate file on: 09/15/2023, 10:31:04
// Original sampling rate fs =  20.00 kHz
// Size of input array: 40000
// ---------------------------------------------------
// Code Example for Implementation in Testbench:
//
//	NeuralSim_Mem DataSim(
//		.CLK_ADC(clk0),
//		.nRST(reset_n),
//		.EN(enable_sim)
//		.DATA_OUT(data_sim),
//		.DATA_END(sim_end)
//	);
// ---------------------------------------------------

module NeuralSim_Mem(
	input wire CLK_ADC,
	input wire nRST,
	input wire EN,
	output wire signed [11:0] DATA_OUT,
	output wire DATA_END
);

reg [15:0] cnt_pos;

reg signed [11:0] bram_data [0:39999];
assign DATA_OUT = (EN && !DATA_END) ? bram_data[cnt_pos] : 12'd0;

assign DATA_END = (cnt_pos == 'd39999);

initial begin
	$readmemh("data_raw.mem", bram_data);
end

always@(posedge CLK_ADC or negedge nRST) begin
	if(!nRST) begin
		cnt_pos = 16'd0;
	end else begin
		cnt_pos = cnt_pos + ((EN && !DATA_END) ? 16'd1 : 16'd0);
	end
end

endmodule
