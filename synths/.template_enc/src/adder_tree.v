`define READ_BASE 2*N - (2*N >> (level - 1))
`define WRITE_BASE 2*N - (N >> (level - 1))

module adder_tree #(N = 16) (
    input [N*32 -1 : 0] inputs,
    output [31:0] out
);

    genvar level, i;
    wire [31:0] inter_stage_sum [2*N-1:0];
    generate
        // Instantiate 32-bit ripple-carry adders for each stage
        for (level = 0; level < $clog2(N) + 1; level = level + 1) begin : adder_tree_level
            if (level == 0) begin
                for (i = 0; i < N; i = i + 1) begin
                    // First stage - directly connect inputs to inter-stage sums
                    assign inter_stage_sum[i] = inputs[i*32+:32];
                end
            end else begin
                for (i = 0; i < (N >> level); i++) begin
                    assign inter_stage_sum[`WRITE_BASE + i] = inter_stage_sum[`READ_BASE + 2*i] + inter_stage_sum[`READ_BASE + 2*i + 1];
                end
            end
        end
    endgenerate
    assign out = inter_stage_sum[2*N-2];

endmodule
