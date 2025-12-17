// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from dsr_msgs2:srv/GetCurrentRotm.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "dsr_msgs2/srv/detail/get_current_rotm__struct.h"
#include "dsr_msgs2/srv/detail/get_current_rotm__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool dsr_msgs2__srv__get_current_rotm__request__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[55];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("dsr_msgs2.srv._get_current_rotm.GetCurrentRotm_Request", full_classname_dest, 54) == 0);
  }
  dsr_msgs2__srv__GetCurrentRotm_Request * ros_message = _ros_message;
  {  // ref
    PyObject * field = PyObject_GetAttrString(_pymsg, "ref");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->ref = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * dsr_msgs2__srv__get_current_rotm__request__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GetCurrentRotm_Request */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("dsr_msgs2.srv._get_current_rotm");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GetCurrentRotm_Request");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  dsr_msgs2__srv__GetCurrentRotm_Request * ros_message = (dsr_msgs2__srv__GetCurrentRotm_Request *)raw_ros_message;
  {  // ref
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->ref);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ref", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// already included above
// #include <Python.h>
// already included above
// #include <stdbool.h>
// already included above
// #include "numpy/ndarrayobject.h"
// already included above
// #include "rosidl_runtime_c/visibility_control.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_current_rotm__struct.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_current_rotm__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "std_msgs/msg/detail/float64_multi_array__functions.h"
// end nested array functions include
ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__float64_multi_array__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__float64_multi_array__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool dsr_msgs2__srv__get_current_rotm__response__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[56];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("dsr_msgs2.srv._get_current_rotm.GetCurrentRotm_Response", full_classname_dest, 55) == 0);
  }
  dsr_msgs2__srv__GetCurrentRotm_Response * ros_message = _ros_message;
  {  // rot_matrix
    PyObject * field = PyObject_GetAttrString(_pymsg, "rot_matrix");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'rot_matrix'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&(ros_message->rot_matrix), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create std_msgs__msg__Float64MultiArray__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    std_msgs__msg__Float64MultiArray * dest = ros_message->rot_matrix.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!std_msgs__msg__float64_multi_array__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // success
    PyObject * field = PyObject_GetAttrString(_pymsg, "success");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->success = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * dsr_msgs2__srv__get_current_rotm__response__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of GetCurrentRotm_Response */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("dsr_msgs2.srv._get_current_rotm");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "GetCurrentRotm_Response");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  dsr_msgs2__srv__GetCurrentRotm_Response * ros_message = (dsr_msgs2__srv__GetCurrentRotm_Response *)raw_ros_message;
  {  // rot_matrix
    PyObject * field = NULL;
    size_t size = ros_message->rot_matrix.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    std_msgs__msg__Float64MultiArray * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->rot_matrix.data[i]);
      PyObject * pyitem = std_msgs__msg__float64_multi_array__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "rot_matrix", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // success
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->success ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "success", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
