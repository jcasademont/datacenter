var width = 960, height = 500;

// create module for custom directives
var graphApp = angular.module('d3DemoApp', []);

// controller business logic
graphApp.controller('AppCtrl', function AppCtrl ($scope, $http) {

    $scope.data = {"nodes": [], "links": []};
    var rescale = function (data) {

        nodes = data.nodes
        links = data.links

        for (var i=0; i < nodes.length; i++) {

            nodes[i].x *= width;
            nodes[i].y *= height;

        }

        for (var i=0; i < links.length; i++) {

            links[i].source.x *= width;
            links[i].source.y *= height;

            links[i].target.x *= width;
            links[i].target.y *= height;

        }

        return {nodes, links};
    }

    $http.get('graph.json')
        .success(
            function(data) {
                console.log("Success");
                $scope.data = rescale(data);
            }
        )
        .error(
            function(data, status) {
                $scope.error = 'Error: ' + status;
            }
        );
});

graphApp.directive('ghVisualization', function () {

    var color = d3.scale.category20();

    var fisheye = d3.fisheye.circular()
                            .radius(200)
                            .distortion(2)

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
                      .style("border", "1px solid black")
                      .attr('preserveAspectRatio', 'xMinYMin slice');

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
                          .attr("x1",
                                  function(d) {
                                      return d.source.x; })
                          .attr("y1",
                                  function(d) {
                                      return d.source.y; })
                          .attr("x2",
                                  function(d) {
                                      return d.target.x; })
                          .attr("y2",
                                  function(d) {
                                      return d.target.y; })
                          .style("stroke-width",
                                  function(d) {
                                      return Math.sqrt(d.value); });

            var node = svg.selectAll(".node")
                          .data(newG.nodes)
                          .enter().append("circle")
                          .attr("class", "node")
                          .attr("r", 5)
                          .attr("cx",
                                  function(d) {
                                      return d.x; })
                          .attr("cy",
                                  function(d) {
                                      return d.y; })
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

            svg.on("mousemove", function() {
              fisheye.focus(d3.mouse(this));

              node.each(function(d) { d.fisheye = fisheye(d); })
                  .attr("cx", function(d) { return d.fisheye.x; })
                  .attr("cy", function(d) { return d.fisheye.y; })
                  .attr("r",
                     function(d) { return d.fisheye.z * 4.5; });

              link.attr("x1",
                     function(d) { return d.source.fisheye.x; })
                  .attr("y1",
                     function(d) { return d.source.fisheye.y; })
                  .attr("x2",
                     function(d) { return d.target.fisheye.x; })
                  .attr("y2",
                     function(d) { return d.target.fisheye.y; });
            })

          });
        }
    };
});
