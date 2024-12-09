module weight_mux #(OUTPUTS, OUTPUT_SIZE) (
    z_out,
    o_out,
    W,
    adder_in
);

    input signed [OUTPUTS*32 - 1:0] W;
    output signed [OUTPUT_SIZE*32 - 1:0] adder_in;
    input [OUTPUTS - 1:0] z_out, o_out;

    genvar i;
    generate
        for(i = 0; i < OUTPUTS; i = i + 1) begin
            assign adder_in[i*32+:32] = (z_out[i] ^ o_out[i]) ? ((z_out[i]) ? W[i*32+:32]: -W[i*32+:32]):0;
        end
        for(i = OUTPUTS; i < OUTPUT_SIZE; i = i + 1) begin
            assign adder_in[i*32+:32] = 0;
        end
    endgenerate
endmodule
