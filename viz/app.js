// create module for custom directives
var graphApp = angular.module('d3DemoApp', []);

// controller business logic
graphApp.controller('AppCtrl', function AppCtrl ($scope, $http) {

    $scope.data = {};
    $http.get('graph.json')
        .success(
            function(data) {
                console.log("Success");
                $scope.data = data;
            }
        )
        .error(
            function(data, status) {
                $scope.error = 'Error: ' + status;
            }
        );
});

graphApp.directive('ghVisualization', function () {

    var width = 960,
        height = 500;

    var color = d3.scale.category20();

    return {
        restrict: 'E',
        terminal: true,
        scope: {
            val: '=',
        },
        link: function (scope, element, attrs) {

          var force = d3.layout.force()
                      .charge(-120)
                      .linkDistance(30)
                      .size([width, height]);

          var svg = d3.select(element[0])
                      .append("svg")
                      .attr("width", width)
                      .attr("height", height)

          scope.$watch('val', function (newG, oldG) {

            if(!newG) {
                return;
            }

            force.nodes(newG.nodes)
                 .links(newG.links)
                 .start();

            var link = svg.selectAll(".link")
                          .data(newG.links)
                          .enter().append("line")
                          .attr("class", "link")
                          .style("stroke-width",
                            function(d) { return Math.sqrt(d.value); });

            var node = svg.selectAll(".node")
                          .data(newG.nodes)
                          .enter().append("circle")
                          .attr("class", "node")
                          .attr("r", 5)
                          .style("fill",
                              function(d) { return color(d.group); })
                          .call(force.drag);

            node.append("title")
                  .text(function(d) { return d.name; });

            force.on("tick", function() {
              link.attr("x1", function(d) { return d.source.x; })
                  .attr("y1", function(d) { return d.source.y; })
                  .attr("x2", function(d) { return d.target.x; })
                  .attr("y2", function(d) { return d.target.y; });

              node.attr("cx", function(d) { return d.x; })
                  .attr("cy", function(d) { return d.y; });
            });

          });
        }
    };
});
