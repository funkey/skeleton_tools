__author__ = 'julia'

from xml.dom import minidom
import numpy as np
import networkx as nx
import skeleton_tools



def from_nx_graphs_to_knossos(nx_skeletons, filename, comment=False):
    doc = to_xml_string(nx_skeletons, comment=comment)

    try:
        f = open(filename, "w")
        f.write(doc)
        f.close()
        print("file written to ", filename)
    except:
        print("Couldn't open file for writing.")


def to_xml_string(nx_skeletons, comment=False):

    doc = minidom.Document()
    annotations_elem = doc.createElement("things")
    doc.appendChild(annotations_elem)
    comments_element = doc.createElement('comments')
    annotations_elem.appendChild(comments_element)
    id_counter = 1  # Although there are different "things" in knossos,
    # the node ids have to be unique within one knossos nml file.
    for skeleton_id, skeleton in enumerate(nx_skeletons):
        if skeleton.name != '' and not comment:
            comment = skeleton.name

        g = nx.convert_node_labels_to_integers(skeleton)
        doc = skeleton_to_nml(doc, annotations_elem, skeleton_id + 1, g, comments_element, node_id_offset=id_counter,
                              comment=comment)
        id_counter += g.number_of_nodes() + 1
        comment = False
    return doc.toprettyxml()


def skeleton_to_nml(doc, annotations_elem, annotation_ID, skeleton, comments_element, node_id_offset=0, comment=False,
                    add_coord_num=0):
    annotation_elem = doc.createElement("thing")
    build_attributes(annotation_elem, [["id", annotation_ID]])
    if comment == 'seg_id':
        if 'seg_id' in skeleton.graph:
            seg_id = skeleton.graph['seg_id']
            build_attributes(annotation_elem, [["comment", 'seg_id %i' % seg_id]])
        else:
            'no seg id in graph dictionary'
    if comment == 'winner':
        build_attributes(annotation_elem, [["comment", 'winner']])

    if comment == 'looser':
        build_attributes(annotation_elem, [["comment", 'looser']])

    nodes_elem = doc.createElement("nodes")
    edges_elem = doc.createElement("edges")

    for node_id, feature_dic in skeleton.nodes_iter(data=True):
        node_elem = doc.createElement("node")
        # node_pos = np.array(feature_dic['position']).astype(np.int)
        node_pos = feature_dic['position'].voxel.astype(np.int)

        if 'radius' in feature_dic:
            radius = feature_dic['radius']
        else:
            radius = 0.5
        # Knossos starts counting at one
        build_attributes(node_elem, [['x', node_pos[0] + add_coord_num], ['y', node_pos[1] + add_coord_num],
                                     ['z', node_pos[2] + add_coord_num],
                                     ['id', node_id + 1 + node_id_offset], ['radius', radius], ['inVp', 0], ['inMag', 0], ['time','123']]
                                    ) # x, y, z, inVp, inMag, time, ID, radius
        if 'comment' in feature_dic:
            node_comment = feature_dic['comment']
            comment_element = doc.createElement('comment')
            build_attributes(comment_element, [['node', node_id + 1 + node_id_offset],
                                               ['content', node_comment]])
            comments_element.appendChild(comment_element)

        nodes_elem.appendChild(node_elem)

    for edge in skeleton.edges_iter():
        edge_elem = doc.createElement("edge")
        build_attributes(edge_elem,
                         [["source", edge[0] + 1 + node_id_offset], ["target", edge[1] + 1 + node_id_offset]])
        edges_elem.appendChild(edge_elem)
    annotation_elem.appendChild(nodes_elem)
    annotation_elem.appendChild(edges_elem)
    annotations_elem.appendChild(annotation_elem)
    # annotation_elem.appendChild(comments_element)
    return doc


def build_attributes(xml_elem, attributes):
    for attr in attributes:
        try:
            xml_elem.setAttribute(attr[0], str(attr[1]))
        except UnicodeEncodeError:
            xml_elem.setAttribute(attr[0], str(attr[1].encode('ascii', 'replace')))
    return


def from_nml_to_nx_skeletons(filename, scaling=[1, 1, 1]):
    doc = minidom.parse(filename)
    annotation_elems = doc.getElementsByTagName("thing")
    nx_skeleton_list = []
    for annotation_elem in annotation_elems:
        nx_skeleton = from_thing_to_nx_skeleton(annotation_elem)
        nx_skeleton_list.append(nx_skeleton)
    return nx_skeleton_list


def from_thing_to_nx_skeleton(annotation_elem):

    # Read nodes
    node_elems = annotation_elem.getElementsByTagName("node")
    point_dic = {}
    for node_elem in node_elems:
        point, id = from_node_elem_to_node(node_elem)
        point = point
        if id in point_dic:
            print('Warning: ID already exists')
        else:
            point_dic[id] = point

    # Read edges
    edge_list = []
    edge_elems = annotation_elem.getElementsByTagName("edge")
    for edge_elem in edge_elems:
        (source_ID, target_ID) = parse_attributes(edge_elem, [["source", int], ["target", int]])
        edge_list.append([source_ID, target_ID])

    skeleton = skeleton_tools.Skeleton()
    skeleton.initialize_from_datapoints(point_dic, True, edgelist=edge_list, datapoints_type='dic')

    return skeleton


def from_node_elem_to_node(node_elem):
    [x, y, z, inVp, inMag, time, ID, radius] = parse_attributes(node_elem, \
                                                                [["x", int], ["y", int], ["z", int], ["inVp", int],
                                                                 ["inMag", int],
                                                                 ["time", int], ["id", int], ["radius", float]])
    point = np.array([x, y, z])
    return point, ID


def fromNml(self, annotation, node_elem):
    self.resetObject()
    self.annotation = annotation
    [x, y, z, inVp, inMag, time, ID, radius] = parse_attributes(node_elem, \
                                                                [["x", int], ["y", int], ["z", int], ["inVp", int],
                                                                 ["inMag", int],
                                                                 ["time", int], ["id", int], ["radius", float]])
    self.ID = ID
    self.x = x
    self.y = y
    self.z = z
    try:
        self.x_scaled = self.x * self.annotation.scaling[0]
        self.y_scaled = self.y * self.annotation.scaling[1]
        self.z_scaled = self.z * self.annotation.scaling[2]
    except TypeError:
        self.x_scaled = self.x
        self.y_scaled = self.y
        self.z_scaled = self.z

    self.setDataElem("inVp", inVp)
    self.setDataElem("radius", radius)
    self.setDataElem("inMag", inMag)
    self.setDataElem("time", time)
    return self


def parse_attributes(xml_elem, parse_input):
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            parse_output.append(x[1](attributes[x[0]].value))
        except KeyError:
            parse_output.append(None)
    return parse_output
