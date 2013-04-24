cpp_stimulus = """
class Stimulus : public Expression
{
public:

  boost::shared_ptr<MeshFunction<std::size_t> > cell_data;

  Stimulus() : Expression(), amplitude(0), duration(0), t(0)
  {
  }

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    assert(cell_data);
    const Cell cell(cell_data->mesh(), c.index);
    switch ((*cell_data)[cell.index()])
    {
    case 0:
      values[0] = 0.0;
      break;
    case 1:
      if (t <= duration)
        values[0] = amplitude;
      else
        values[0] = 0.0;
      break;
    default:
      values[0] = 0.0;
    }
  }
  double amplitude;
  double duration;
  double t;
};"""

